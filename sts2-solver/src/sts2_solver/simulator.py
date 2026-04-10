"""Act 1 run simulator — pure algorithmic, no LLMs.

Simulates complete Act 1 (Overgrowth) runs using:
- Existing combat engine + solver for card play optimization
- Probabilistic enemy AI derived from monster data
- Card reward pools with rarity weighting
- Rest sites, events, and a simple map model

Usage:
    python -m sts2_solver.simulator --runs 1000 --character ironclad
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .combat_engine import (
    can_play_card,
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_turn,
    tick_enemy_powers,
)
from .config import EVALUATOR, CARD_TIERS, STRATEGY
from .constants import CardType, TargetType
from .data_loader import CardDB, load_cards, DEFAULT_DATA_DIR
from .evaluator import evaluate_turn
from .models import Card, CombatState, EnemyState, PlayerState
from .solver import solve_turn


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_json(filename: str) -> Any:
    path = DEFAULT_DATA_DIR / filename
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_MONSTERS_BY_ID: dict[str, dict] = {}
_ENCOUNTERS_BY_ID: dict[str, dict] = {}
_EVENTS_BY_ID: dict[str, dict] = {}
_RELICS_BY_ID: dict[str, dict] = {}
_ACTS_BY_ID: dict[str, dict] = {}
_CHARACTERS_BY_ID: dict[str, dict] = {}


def _ensure_data_loaded():
    if _MONSTERS_BY_ID:
        return
    for m in _load_json("monsters.json"):
        _MONSTERS_BY_ID[m["id"]] = m
    for e in _load_json("encounters.json"):
        _ENCOUNTERS_BY_ID[e["id"]] = e
    for ev in _load_json("events.json"):
        _EVENTS_BY_ID[ev["id"]] = ev
    for r in _load_json("relics.json"):
        _RELICS_BY_ID[r["id"]] = r
    for a in _load_json("acts.json"):
        _ACTS_BY_ID[a["id"]] = a
    for c in _load_json("characters.json"):
        _CHARACTERS_BY_ID[c["id"]] = c


# ---------------------------------------------------------------------------
# Card ID normalization: characters.json uses "StrikeIronclad" but
# cards.json uses "STRIKE_IRONCLAD"
# ---------------------------------------------------------------------------

def _normalize_card_id(raw_id: str) -> str:
    """Convert camelCase card IDs to UPPER_SNAKE_CASE."""
    # Insert underscore before uppercase letters, then uppercase all
    import re
    result = re.sub(r'(?<=[a-z])(?=[A-Z])', '_', raw_id)
    return result.upper()


# ---------------------------------------------------------------------------
# Enemy AI: probabilistic move selection with mechanical effects
# ---------------------------------------------------------------------------

# Hand-coded intent data for Act 1 (Overgrowth) enemies.
# Format: list of (intent_type, damage, hits, block, buff_effects)
# buff_effects: dict of effects to apply, e.g. {"self_strength": 2}
#
# Derived from monsters.json move lists + damage tables + STS conventions.
# Enemies cycle through their moves, which produces realistic patterns.

ENEMY_MOVE_TABLES: dict[str, list[dict]] = {
    # --- Weak encounters ---
    "NIBBIT": [
        {"type": "Attack", "damage": 12, "hits": 1},       # Butt
        {"type": "Attack", "damage": 6, "hits": 2},         # Slice x2
        {"type": "Buff", "self_strength": 2, "self_block": 5},  # Hiss
    ],
    "SHRINKER_BEETLE": [
        {"type": "Debuff", "player_shrink": 1},                   # Shrinker (applies -1 Strength via Shrink)
        {"type": "Attack", "damage": 7, "hits": 1},          # Chomp
        {"type": "Attack", "damage": 13, "hits": 1},         # Stomp
    ],
    "FUZZY_WURM_CRAWLER": [
        {"type": "Debuff", "player_frail": 1, "damage": 4},  # Acid Goop (debuff+damage)
        {"type": "Attack", "damage": 4, "hits": 1},          # Acid Goop
        {"type": "Buff", "self_strength": 3},                 # Inhale (charge up)
    ],

    # --- Normal encounters ---
    "FLYCONID": [
        {"type": "Debuff", "player_vulnerable": 2},                             # Weakening Spores (no dmg)
        {"type": "Attack", "damage": 8, "hits": 1, "player_frail": 2},       # Frail Spores + dmg
        {"type": "Attack", "damage": 11, "hits": 1},                          # Smash
    ],
    "FOGMOG": [
        {"type": "Buff"},                                                   # Illusory Spores (summons Eye)
        {"type": "Attack", "damage": 8, "hits": 1, "self_strength": 1},   # Thwack (dmg + gain 1 STR)
        {"type": "Attack", "damage": 14, "hits": 1},                      # Headbutt
    ],
    "EYE_WITH_TEETH": [
        {"type": "StatusCard"},                                            # Distract (adds 3 Dazed)
    ],
    "CUBEX_CONSTRUCT": [
        {"type": "Buff", "self_strength": 2},                # Charge Up
        {"type": "Attack", "damage": 5, "hits": 2},          # Repeater x2
        {"type": "Attack", "damage": 5, "hits": 3},          # Repeater x3
        {"type": "Attack", "damage": 7, "hits": 1},          # Expel Blast
        {"type": "Defend", "block": 12},                      # Submerge
    ],
    "MAWLER": [
        {"type": "Attack", "damage": 14, "hits": 1},         # Rip and Tear
        {"type": "Buff", "self_strength": 3},                 # Roar
        {"type": "Attack", "damage": 4, "hits": 3},          # Claw x3
    ],
    "VINE_SHAMBLER": [
        {"type": "Attack", "damage": 6, "hits": 2},                               # Swipe x2
        {"type": "Attack", "damage": 8, "hits": 1, "player_tangled": 2},          # Grasping Vines (2 turns)
        {"type": "Attack", "damage": 16, "hits": 1},                              # Chomp
    ],
    "SLITHERING_STRANGLER": [
        {"type": "Debuff", "player_constrict": 3, "self_block": 5},   # Constrict (+ block)
        {"type": "Attack", "damage": 7, "hits": 1, "self_block": 5},  # Thwack (dmg + block)
        {"type": "Attack", "damage": 12, "hits": 1},                  # Lash
    ],
    "SNAPPING_JAXFRUIT": [
        {"type": "Attack", "damage": 3, "hits": 1, "self_strength": 2},  # Energy Orb (dmg + gain 2 STR)
    ],
    "INKLET": [
        {"type": "Attack", "damage": 3, "hits": 1},          # Jab
        {"type": "Attack", "damage": 2, "hits": 3},          # Whirlwind x3
        {"type": "Attack", "damage": 10, "hits": 1},         # Piercing Gaze
    ],

    # Slimes — debuffs are single-turn applications
    "LEAF_SLIME_M": [
        {"type": "Attack", "damage": 8, "hits": 1},          # Clump Shot
        {"type": "Attack", "damage": 8, "hits": 1},          # Clump Shot again
        {"type": "Debuff", "player_frail": 1},                # Sticky Shot
    ],
    "LEAF_SLIME_S": [
        {"type": "Attack", "damage": 3, "hits": 1},          # Butt
        {"type": "Attack", "damage": 3, "hits": 1},          # Butt again
        {"type": "Debuff", "player_weak": 1},                 # Goop
    ],
    "TWIG_SLIME_M": [
        {"type": "Attack", "damage": 11, "hits": 1},         # Clump Shot
        {"type": "Attack", "damage": 11, "hits": 1},         # Clump Shot again
        {"type": "Debuff", "player_vulnerable": 1},           # Sticky Shot
    ],
    "TWIG_SLIME_S": [
        {"type": "Attack", "damage": 4, "hits": 1},          # Butt
    ],

    # Ruby Raiders
    "ASSASSIN_RUBY_RAIDER": [
        {"type": "Attack", "damage": 11, "hits": 1},         # Killshot
    ],
    "AXE_RUBY_RAIDER": [
        {"type": "Attack", "damage": 5, "hits": 1, "self_block": 5},  # Swing (dmg + block)
        {"type": "Attack", "damage": 5, "hits": 1, "self_block": 5},  # Swing (repeats)
        {"type": "Attack", "damage": 12, "hits": 1, "self_block": 5}, # Big Swing (+ block)
    ],
    "BRUTE_RUBY_RAIDER": [
        {"type": "Attack", "damage": 7, "hits": 1},           # Beat
        {"type": "Buff", "self_strength": 3},                  # Clap (gain 3 STR)
    ],
    "CROSSBOW_RUBY_RAIDER": [
        {"type": "Defend"},                                             # Reload (no block)
        {"type": "Attack", "damage": 14, "hits": 1, "self_block": 3},  # Fire! (+ block)
    ],
    "TRACKER_RUBY_RAIDER": [
        {"type": "Debuff", "player_frail": 2},                # Track (applies 2 Frail)
        {"type": "Attack", "damage": 1, "hits": 8},           # Unleash the Hounds (1x8)
    ],

    # --- Elites ---
    "BYGONE_EFFIGY": [
        {"type": "Buff"},                                     # Initial Sleep (skip)
        {"type": "Buff", "self_strength": 5},                 # Wake (big buff)
        {"type": "Buff", "self_strength": 2},                 # Sleep (gaining power)
        {"type": "Attack", "damage": 15, "hits": 3},         # Slashes x3
    ],
    "BYRDONIS": [
        {"type": "Attack", "damage": 3, "hits": 4},          # Peck x4
        {"type": "Attack", "damage": 16, "hits": 1},         # Swoop
    ],
    "PHROG_PARASITE": [
        {"type": "Debuff", "player_frail": 2, "player_weak": 2},  # Infect
        {"type": "Attack", "damage": 4, "hits": 4},               # Lash x4
    ],

    # --- Bosses ---
    "CEREMONIAL_BEAST": [
        {"type": "Buff", "self_strength": 3, "self_block": 10},    # Beast Cry (buff+block)
        {"type": "Attack", "damage": 18, "hits": 1},               # Plow
        {"type": "Debuff", "player_vulnerable": 2, "player_weak": 2},  # Stun
        {"type": "Attack", "damage": 15, "hits": 2},               # Stomp x2
        {"type": "Attack", "damage": 17, "hits": 1},               # Crush
        {"type": "Buff", "self_strength": 4, "self_block": 15},    # Beast Cry (stronger)
    ],
    "VANTOM": [
        {"type": "Attack", "damage": 7, "hits": 2},                # Ink Blot x2
        {"type": "Attack", "damage": 6, "hits": 3},                # Inky Lance x3
        {"type": "Buff", "self_strength": 4},                       # Prepare
        {"type": "Attack", "damage": 27, "hits": 1},               # Dismember
    ],
    "KIN_FOLLOWER": [
        {"type": "Attack", "damage": 5, "hits": 2},                # Quick Slash x2
        {"type": "Attack", "damage": 2, "hits": 4},                # Boomerang x4
        {"type": "Buff", "all_strength": 2},                       # Power Dance (buffs team)
    ],
    "KIN_PRIEST": [
        {"type": "Debuff", "player_frail": 2, "damage": 8},       # Orb Of Frailty
        {"type": "Debuff", "player_weak": 2, "damage": 8},        # Orb Of Weakness
        {"type": "Attack", "damage": 3, "hits": 5},                # Beam x5
        {"type": "Buff", "all_strength": 3},                       # Ritual (buffs team)
    ],

    # ── New weak encounters ──
    "CORPSE_SLUG": [
        {"type": "Attack", "damage": 5, "hits": 2},                # Whip Slap x2
        {"type": "Attack", "damage": 10, "hits": 1},               # Glomp
        {"type": "Debuff", "player_frail": 1},                     # Goop
    ],
    "EXOSKELETON": [
        {"type": "Attack", "damage": 5, "hits": 2},                # Skitter x2
        {"type": "Attack", "damage": 12, "hits": 1},               # Mandible
        {"type": "Buff", "self_strength": 2},                       # Enrage
    ],
    "SCROLL_OF_BITING": [
        {"type": "Attack", "damage": 10, "hits": 1},               # Chomp
        {"type": "Attack", "damage": 4, "hits": 3},                # Chew x3
        {"type": "Buff", "self_strength": 3},                       # More Teeth
    ],
    "SEAPUNK": [
        {"type": "Attack", "damage": 9, "hits": 1},                # Sea Kick
        {"type": "Attack", "damage": 6, "hits": 2},                # Spinning Kick x2
        {"type": "Debuff", "player_weak": 1, "self_block": 6},    # Bubble Burp
    ],
    "SLUDGE_SPINNER": [
        {"type": "Debuff", "player_frail": 1, "player_weak": 1},  # Oil Spray
        {"type": "Attack", "damage": 12, "hits": 1},               # Slam
        {"type": "Buff", "self_strength": 2},                       # Rage
    ],
    "TUNNELER": [
        {"type": "Attack", "damage": 8, "hits": 1},                # Bite
        {"type": "Defend", "block": 10},                            # Burrow (digs down)
        {"type": "Attack", "damage": 14, "hits": 1},               # Below Move (emerges)
        {"type": "Debuff", "player_weak": 1},                      # Dizzy
    ],
    "TOADPOLE": [
        {"type": "Attack", "damage": 6, "hits": 1},                # Spike Spit
        {"type": "Attack", "damage": 3, "hits": 3},                # Whirl x3
        {"type": "Buff", "self_strength": 2, "self_block": 4},    # Spiken
    ],
    "THIEVING_HOPPER": [
        {"type": "Attack", "damage": 6, "hits": 1},                # Thievery (steals gold)
        {"type": "Attack", "damage": 8, "hits": 1},                # Nab
        {"type": "Attack", "damage": 4, "hits": 3},                # Hat Trick x3
        {"type": "Defend", "block": 8},                             # Flutter
    ],
    "DEVOTED_SCULPTOR": [
        {"type": "Buff", "self_strength": 4},                       # Forbidden Incantation
        {"type": "Attack", "damage": 16, "hits": 1},               # Savage
    ],
    "WRIGGLER": [
        {"type": "Attack", "damage": 7, "hits": 1},                # Nasty Bite
        {"type": "Attack", "damage": 4, "hits": 2},                # Wriggle x2
    ],

    # ── New normal encounters ──
    "CHOMPER": [
        {"type": "Attack", "damage": 8, "hits": 1, "self_block": 4},  # Clamp (dmg + block)
        {"type": "Attack", "damage": 8, "hits": 1, "self_block": 4},  # Clamp repeats
        {"type": "Debuff", "player_vulnerable": 2},                    # Screech
    ],
    "BOWLBUG_EGG": [
        {"type": "Attack", "damage": 6, "hits": 1},                # Bite
    ],
    "BOWLBUG_NECTAR": [
        {"type": "Attack", "damage": 7, "hits": 1},                # Thrash
        {"type": "Buff", "all_strength": 1},                        # Buff (heals/buffs team)
        {"type": "Attack", "damage": 10, "hits": 1},               # Thrash (stronger)
    ],
    "BOWLBUG_ROCK": [
        {"type": "Attack", "damage": 14, "hits": 1},               # Headbutt
        {"type": "Debuff", "player_weak": 1},                      # Dizzy
    ],
    "BOWLBUG_SILK": [
        {"type": "Attack", "damage": 9, "hits": 1},                # Trash
        {"type": "Debuff", "player_frail": 2, "damage": 5},       # Toxic Spit
    ],
    "TWO_TAILED_RAT": [
        {"type": "Attack", "damage": 5, "hits": 2},                # Scratch x2
        {"type": "Attack", "damage": 8, "hits": 1, "player_frail": 1},  # Disease Bite
        {"type": "Debuff", "player_weak": 1},                      # Screech
    ],
    "PUNCH_CONSTRUCT": [
        {"type": "Buff", "self_strength": 3},                       # Ready (charge up)
        {"type": "Attack", "damage": 18, "hits": 1},               # Strong Punch
        {"type": "Attack", "damage": 6, "hits": 3},                # Fast Punch x3
    ],
    "FROG_KNIGHT": [
        {"type": "Buff", "self_strength": 2, "self_block": 8},    # For the Queen
        {"type": "Attack", "damage": 16, "hits": 1},               # Strike Down Evil
        {"type": "Attack", "damage": 5, "hits": 3},                # Tongue Lash x3
        {"type": "Attack", "damage": 20, "hits": 1},               # Beetle Charge
    ],
    "FOSSIL_STALKER": [
        {"type": "Attack", "damage": 10, "hits": 1},               # Tackle
        {"type": "Debuff", "player_vulnerable": 2, "damage": 6},  # Latch
        {"type": "Attack", "damage": 4, "hits": 4},                # Lash x4
    ],
    "SPINY_TOAD": [
        {"type": "Defend", "block": 10},                            # Protruding Spikes (thorns)
        {"type": "Attack", "damage": 6, "hits": 4},                # Spike Explosion x4
        {"type": "Attack", "damage": 14, "hits": 1},               # Tongue Lash
    ],
    "LIVING_FOG": [
        {"type": "Debuff", "player_frail": 1, "player_weak": 1},  # Advanced Gas
        {"type": "Buff", "self_strength": 3},                       # Bloat
        {"type": "Attack", "damage": 18, "hits": 1},               # Super Gas Blast
    ],
    "GAS_BOMB": [
        {"type": "Attack", "damage": 20, "hits": 1},               # Explode (dies after)
    ],
    "LOUSE_PROGENITOR": [
        {"type": "Attack", "damage": 8, "hits": 2},                # Web Cannon x2
        {"type": "Attack", "damage": 14, "hits": 1},               # Pounce
        {"type": "Buff", "self_strength": 3, "self_block": 10},   # Curl and Grow
    ],
    "HUNTER_KILLER": [
        {"type": "Debuff", "player_vulnerable": 2},                # Tenderizing Goop
        {"type": "Attack", "damage": 12, "hits": 1},               # Bite
        {"type": "Attack", "damage": 5, "hits": 3},                # Puncture x3
    ],
    "FABRICATOR": [
        {"type": "Buff", "self_strength": 2, "self_block": 6},    # Fabricate
        {"type": "Attack", "damage": 10, "hits": 1},               # Fabricating Strike
        {"type": "Attack", "damage": 22, "hits": 1},               # Disintegrate
    ],
    "CALCIFIED_CULTIST": [
        {"type": "Buff", "self_strength": 2},                       # Ritual
        {"type": "Attack", "damage": 10, "hits": 1},               # Smash
        {"type": "Attack", "damage": 6, "hits": 2},                # Dark Strike x2
    ],
    "DAMP_CULTIST": [
        {"type": "Debuff", "player_weak": 2},                      # Hex
        {"type": "Attack", "damage": 8, "hits": 1},                # Chop
        {"type": "Buff", "all_strength": 1},                        # Incantation
    ],
    "OWL_MAGISTRATE": [
        {"type": "Debuff", "player_frail": 2, "player_weak": 1},  # Judgement
        {"type": "Attack", "damage": 10, "hits": 2},               # Talon Strike x2
        {"type": "Buff", "self_strength": 3, "self_block": 8},    # Roost
    ],
    "SLIMED_BERSERKER": [
        {"type": "Attack", "damage": 7, "hits": 1},                # Slime Attack
        {"type": "Buff", "self_strength": 4},                       # Rage
        {"type": "Attack", "damage": 5, "hits": 3},                # Flurry x3
    ],
    "MYTE": [
        {"type": "Attack", "damage": 4, "hits": 1},                # Nibble
        {"type": "Buff", "self_strength": 1},                       # Swarm
    ],
    "AXEBOT": [
        {"type": "Attack", "damage": 10, "hits": 1, "self_block": 5},  # Axe Swing
        {"type": "Attack", "damage": 6, "hits": 2},                     # Double Chop x2
        {"type": "Buff", "self_strength": 2},                            # Sharpen
    ],
    "GLOBE_HEAD": [
        {"type": "Attack", "damage": 8, "hits": 1},                # Beam
        {"type": "Debuff", "player_vulnerable": 1, "damage": 6},  # Flash
        {"type": "Attack", "damage": 12, "hits": 1},               # Overload
    ],
    "HAUNTED_SHIP": [
        {"type": "Attack", "damage": 6, "hits": 3},                # Broadside x3
        {"type": "Debuff", "player_frail": 2},                     # Ghost Wind
        {"type": "Attack", "damage": 18, "hits": 1},               # Ram
    ],
    "SEWER_CLAM": [
        {"type": "Defend", "block": 8},                             # Shell Up
        {"type": "Attack", "damage": 10, "hits": 1, "player_weak": 1},  # Spit
        {"type": "Attack", "damage": 14, "hits": 1},               # Snap
    ],
    "THE_LOST": [
        {"type": "Attack", "damage": 6, "hits": 2},                # Slash x2
        {"type": "Debuff", "player_frail": 1},                     # Haunt
    ],
    "THE_FORGOTTEN": [
        {"type": "Attack", "damage": 12, "hits": 1},               # Crush
        {"type": "Buff", "self_strength": 2, "self_block": 6},    # Remember
    ],
    "THE_OBSCURA": [
        {"type": "Debuff", "player_weak": 2, "player_frail": 1},  # Obscure
        {"type": "Attack", "damage": 9, "hits": 2},                # Shadow Strike x2
        {"type": "Attack", "damage": 16, "hits": 1},               # Void Blast
    ],
    "OVICOPTER": [
        {"type": "Attack", "damage": 8, "hits": 1},                # Swoop
        {"type": "Buff"},                                           # Lay Egg (summons)
        {"type": "Attack", "damage": 5, "hits": 3},                # Barrage x3
    ],
    "TOUGH_EGG": [
        {"type": "Defend", "block": 6},                             # Harden
        {"type": "Attack", "damage": 8, "hits": 1},                # Hatch (emerges)
    ],

    # ── New elites ──
    "DECIMILLIPEDE_SEGMENT_FRONT": [
        {"type": "Attack", "damage": 6, "hits": 3},                # Mandible Flurry x3
        {"type": "Buff", "self_strength": 2, "self_block": 8},    # Burrow
    ],
    "DECIMILLIPEDE_SEGMENT_MIDDLE": [
        {"type": "Attack", "damage": 8, "hits": 2},                # Body Slam x2
        {"type": "Defend", "block": 12},                            # Curl Up
    ],
    "DECIMILLIPEDE_SEGMENT_BACK": [
        {"type": "Attack", "damage": 5, "hits": 4},                # Tail Whip x4
        {"type": "Debuff", "player_vulnerable": 2},                # Acid Spray
    ],
    "ENTOMANCER": [
        {"type": "Buff", "self_strength": 3},                       # Summon Swarm
        {"type": "Attack", "damage": 7, "hits": 3},                # Bug Barrage x3
        {"type": "Debuff", "player_frail": 2, "damage": 10},      # Parasite
        {"type": "Attack", "damage": 20, "hits": 1},               # Devour
    ],
    "SKULKING_COLONY": [
        {"type": "Attack", "damage": 4, "hits": 5},                # Swarm x5
        {"type": "Buff", "self_strength": 3, "self_block": 10},   # Regroup
        {"type": "Attack", "damage": 18, "hits": 1},               # Colony Crush
        {"type": "Debuff", "player_weak": 2, "player_frail": 1},  # Overwhelm
    ],
    "MECHA_KNIGHT": [
        {"type": "Attack", "damage": 12, "hits": 1, "self_block": 8},  # Shield Bash
        {"type": "Attack", "damage": 8, "hits": 3},                     # Triple Strike x3
        {"type": "Buff", "self_strength": 4},                            # Overclock
        {"type": "Attack", "damage": 25, "hits": 1},                    # Mega Slash
    ],
    "INFESTED_PRISM": [
        {"type": "Attack", "damage": 6, "hits": 3},                # Refracted Beam x3
        {"type": "Debuff", "player_vulnerable": 2, "player_weak": 1},  # Prismatic Haze
        {"type": "Attack", "damage": 20, "hits": 1},               # Overcharge
        {"type": "Buff", "self_strength": 3, "self_block": 12},   # Crystal Shell
    ],
    "TERROR_EEL": [
        {"type": "Attack", "damage": 5, "hits": 4},                # Electric Bite x4
        {"type": "Debuff", "player_frail": 2, "player_vulnerable": 2},  # Terrify
        {"type": "Attack", "damage": 22, "hits": 1},               # Thunder Slam
        {"type": "Buff", "self_strength": 4},                       # Charge Up
    ],
    "SOUL_NEXUS": [
        {"type": "Debuff", "player_weak": 3},                      # Soul Drain
        {"type": "Attack", "damage": 8, "hits": 3},                # Spirit Barrage x3
        {"type": "Buff", "self_strength": 5},                       # Absorb
        {"type": "Attack", "damage": 25, "hits": 1},               # Obliterate
    ],
    "PHANTASMAL_GARDENER": [
        {"type": "Debuff", "player_frail": 2},                     # Wilt
        {"type": "Attack", "damage": 10, "hits": 2},               # Vine Lash x2
        {"type": "Buff", "self_strength": 3, "self_block": 10},   # Overgrow
        {"type": "Attack", "damage": 7, "hits": 4},                # Thorn Storm x4
    ],
    "FLAIL_KNIGHT": [
        {"type": "Attack", "damage": 14, "hits": 1},               # Flail Swing
        {"type": "Attack", "damage": 6, "hits": 3},                # Chain Whip x3
        {"type": "Buff", "self_strength": 2, "self_block": 10},   # Rally
    ],
    "MAGI_KNIGHT": [
        {"type": "Debuff", "player_weak": 2, "damage": 8},        # Arcane Bolt
        {"type": "Attack", "damage": 12, "hits": 1},               # Magic Slash
        {"type": "Buff", "all_strength": 2},                        # Empower (team buff)
    ],
    "SPECTRAL_KNIGHT": [
        {"type": "Attack", "damage": 5, "hits": 4},                # Phase Strike x4
        {"type": "Debuff", "player_vulnerable": 2},                # Haunt
        {"type": "Defend", "block": 15},                            # Ethereal Shield
    ],

    # ── New bosses ──
    "DOORMAKER": [
        {"type": "Buff", "self_strength": 3, "self_block": 15},    # Seal Door
        {"type": "Attack", "damage": 10, "hits": 3},               # Door Slam x3
        {"type": "Debuff", "player_frail": 2, "player_weak": 2},  # Dimensional Rip
        {"type": "Attack", "damage": 28, "hits": 1},               # Grand Slam
    ],
    "DOOR": [
        {"type": "Defend", "block": 20},                            # Reinforce
        {"type": "Attack", "damage": 12, "hits": 1},               # Slam
    ],
    "WATERFALL_GIANT": [
        {"type": "Attack", "damage": 8, "hits": 3},                # Cascade x3
        {"type": "Buff", "self_strength": 4, "self_block": 12},   # Rising Tide
        {"type": "Attack", "damage": 30, "hits": 1},               # Tidal Crush
        {"type": "Debuff", "player_frail": 3},                     # Drenching Wave
        {"type": "Attack", "damage": 12, "hits": 3},               # Torrent x3
    ],
    "LAGAVULIN_MATRIARCH": [
        {"type": "Buff"},                                           # Slumber (asleep T1)
        {"type": "Buff"},                                           # Slumber (asleep T2)
        {"type": "Debuff", "player_shrink": 2, "player_frail": 2},  # Wake (debuff burst)
        {"type": "Attack", "damage": 20, "hits": 1},               # Pummel
        {"type": "Attack", "damage": 8, "hits": 3},                # Flurry x3
        {"type": "Buff", "self_strength": 3},                       # Roar
    ],
    "KNOWLEDGE_DEMON": [
        {"type": "Debuff", "player_weak": 2, "player_vulnerable": 2},  # Dark Knowledge
        {"type": "Attack", "damage": 7, "hits": 4},                     # Mind Rend x4
        {"type": "Buff", "self_strength": 5},                            # Study
        {"type": "Attack", "damage": 30, "hits": 1},                    # Enlightened Fury
    ],
    "CRUSHER": [
        {"type": "Attack", "damage": 12, "hits": 2},               # Claw Crush x2
        {"type": "Buff", "self_strength": 3, "self_block": 15},   # Harden Shell
        {"type": "Attack", "damage": 25, "hits": 1},               # Mega Claw
    ],
    "ROCKET": [
        {"type": "Attack", "damage": 5, "hits": 5},                # Rocket Barrage x5
        {"type": "Debuff", "player_vulnerable": 2},                # Lock On
        {"type": "Attack", "damage": 20, "hits": 1},               # Big Shot
    ],
    "QUEEN": [
        {"type": "Buff", "self_strength": 4, "self_block": 12},   # Royal Decree
        {"type": "Attack", "damage": 8, "hits": 4},                # Swarm Command x4
        {"type": "Debuff", "player_frail": 2, "player_weak": 2},  # Weakening Aura
        {"type": "Attack", "damage": 30, "hits": 1},               # Queen's Wrath
    ],
    "TORCH_HEAD_AMALGAM": [
        {"type": "Attack", "damage": 10, "hits": 2},               # Flame Lash x2
        {"type": "Buff", "self_strength": 3},                       # Ignite
        {"type": "Attack", "damage": 18, "hits": 1},               # Fireball
    ],
    "SOUL_FYSH": [
        {"type": "Debuff", "player_weak": 2},                      # Soul Siphon
        {"type": "Attack", "damage": 6, "hits": 4},                # Bubble Barrage x4
        {"type": "Buff", "self_strength": 4, "self_block": 10},   # Deep Dive
        {"type": "Attack", "damage": 25, "hits": 1},               # Leviathan Crush
        {"type": "Debuff", "player_frail": 3, "damage": 12},      # Abyssal Wave
    ],
    "TEST_SUBJECT": [
        {"type": "Buff", "self_strength": 3},                       # Mutate
        {"type": "Attack", "damage": 10, "hits": 2},               # Lash x2
        {"type": "Attack", "damage": 7, "hits": 4},                # Frenzy x4
        {"type": "Debuff", "player_vulnerable": 3},                # Acid Spray
        {"type": "Attack", "damage": 28, "hits": 1},               # Annihilate
    ],
    "THE_INSATIABLE": [
        {"type": "Attack", "damage": 8, "hits": 3},                # Devour x3
        {"type": "Buff", "self_strength": 5},                       # Hunger
        {"type": "Debuff", "player_weak": 2, "player_frail": 2},  # Consume
        {"type": "Attack", "damage": 35, "hits": 1},               # Feast
        {"type": "Attack", "damage": 12, "hits": 3},               # Ravage x3
    ],
}


@dataclass
class EnemyAI:
    """Tracks move cycling for a single enemy instance."""
    monster_id: str
    move_table: list[dict]
    move_index: int = 0

    def pick_intent(self) -> dict:
        """Return the next intent dict.

        Cycles through the hand-coded move table. For enemies without
        a table, falls back to generic data-driven resolution.
        """
        if not self.move_table:
            return {"type": "Attack", "damage": 8, "hits": 1}

        move = self.move_table[self.move_index % len(self.move_table)]
        self.move_index += 1
        return dict(move)  # Copy so caller can mutate


def _create_enemy_ai(monster_id: str) -> EnemyAI:
    """Create an EnemyAI for a monster from data."""
    _ensure_data_loaded()

    # Use hand-coded table if available
    if monster_id in ENEMY_MOVE_TABLES:
        return EnemyAI(
            monster_id=monster_id,
            move_table=ENEMY_MOVE_TABLES[monster_id],
        )

    # Fallback: build a simple table from monsters.json
    monster = _MONSTERS_BY_ID.get(monster_id, {})
    damage_values = monster.get("damage_values") or {}
    moves = monster.get("moves", [])

    table: list[dict] = []
    for move in moves:
        name = move.get("name", "")
        move_id = move.get("id", "")
        damage = _match_damage(name, move_id, damage_values)
        if damage is not None:
            table.append({"type": "Attack", "damage": damage, "hits": 1})
        else:
            # Unknown move — assume light buff
            table.append({"type": "Buff", "self_strength": 1})

    if not table:
        table = [{"type": "Attack", "damage": 8, "hits": 1}]

    return EnemyAI(monster_id=monster_id, move_table=table)


def _match_damage(move_name: str, move_id: str, damage_values: dict) -> int | None:
    """Try to match a move to its damage value."""
    name_lower = move_name.lower().replace(" ", "").replace("_", "")
    id_lower = move_id.lower().replace(" ", "").replace("_", "")
    for key, val in damage_values.items():
        key_lower = key.lower().replace(" ", "").replace("_", "")
        if (key_lower in name_lower or name_lower in key_lower
                or key_lower in id_lower or id_lower in key_lower):
            return val.get("normal", val.get("ascension", 5))
    return None


def _spawn_enemy(monster_id: str) -> EnemyState:
    """Create an EnemyState from monster data."""
    _ensure_data_loaded()
    monster = _MONSTERS_BY_ID.get(monster_id, {})
    min_hp = monster.get("min_hp") or 20
    max_hp = monster.get("max_hp") or min_hp
    hp = random.randint(min_hp, max_hp) if min_hp < max_hp else min_hp
    return EnemyState(
        id=monster_id,
        name=monster.get("name", monster_id),
        hp=hp,
        max_hp=hp,
    )


# ---------------------------------------------------------------------------
# Card reward pool
# ---------------------------------------------------------------------------

# STS-like rarity weights: Common 60%, Uncommon 37%, Rare 3%
RARITY_WEIGHTS = {"Common": 60, "Uncommon": 37, "Rare": 3}
REWARD_CARDS_OFFERED = 3


def _build_card_pool(card_db: CardDB, character_color: str) -> dict[str, list[Card]]:
    """Build card pools grouped by rarity for a character.

    Includes character-specific cards + colorless cards.
    Excludes Basic, Status, Curse, Token, Event, Quest, Ancient.
    """
    pools: dict[str, list[Card]] = {"Common": [], "Uncommon": [], "Rare": []}
    excluded_rarities = {"Basic", "Status", "Curse", "Token", "Event",
                         "Quest", "Ancient"}

    # We need to read raw card data for color/rarity since Card model
    # doesn't store those. Load from JSON directly.
    raw_cards = _load_json("cards.json")
    raw_by_id: dict[str, dict] = {c["id"]: c for c in raw_cards}

    for card in card_db.all_cards():
        if card.upgraded:
            continue
        raw = raw_by_id.get(card.id)
        if raw is None:
            continue
        rarity = raw.get("rarity", "")
        color = raw.get("color", "")
        if rarity in excluded_rarities:
            continue
        if color not in (character_color, "colorless"):
            continue
        if rarity in pools:
            pools[rarity].append(card)

    return pools


def _offer_card_rewards(
    pools: dict[str, list[Card]],
    deck: list[Card],
    count: int = REWARD_CARDS_OFFERED,
) -> list[Card]:
    """Generate a card reward offering (no duplicates, not already in deck)."""
    deck_ids = {c.id for c in deck}
    offered: list[Card] = []
    rarities = list(RARITY_WEIGHTS.keys())
    weights = list(RARITY_WEIGHTS.values())

    attempts = 0
    while len(offered) < count and attempts < 50:
        attempts += 1
        rarity = random.choices(rarities, weights=weights, k=1)[0]
        pool = pools.get(rarity, [])
        if not pool:
            continue
        card = random.choice(pool)
        if card.id not in deck_ids and card.id not in {c.id for c in offered}:
            offered.append(card)
    return offered


# ---------------------------------------------------------------------------
# Algorithmic card pick strategy (no LLM)
# ---------------------------------------------------------------------------

# Build a score map from the tier list
_TIER_SCORES: dict[str, float] = {}


def _init_tier_scores():
    if _TIER_SCORES:
        return
    for card_name in CARD_TIERS.get("S", []):
        _TIER_SCORES[card_name.lower()] = 100
    for card_name in CARD_TIERS.get("A", []):
        _TIER_SCORES[card_name.lower()] = 80
    for card_name in CARD_TIERS.get("B", []):
        _TIER_SCORES[card_name.lower()] = 60
    for card_name in CARD_TIERS.get("avoid", []):
        _TIER_SCORES[card_name.lower()] = -10


def _score_card_for_pick(card: Card, deck: list[Card]) -> float:
    """Score a card for the pick decision. Higher = better to pick.

    Cards NOT in the tier list score 0 (skip by default). Only tier-listed
    cards are considered worth adding. This prevents deck bloat from
    random mediocre commons.
    """
    _init_tier_scores()
    # Unknown cards score 0 — they must be in the tier list to be picked
    score = _TIER_SCORES.get(card.name.lower(), 0)

    # Deck size penalty: progressively harder to justify adding cards
    deck_size = len(deck)
    if deck_size >= STRATEGY["deck_warn_threshold"]:
        score -= 30  # Almost never pick into a bloated deck
    elif deck_size >= STRATEGY["deck_lean_target"]:
        score -= 10

    # Power bonus: scaling cards are very valuable early
    power_count = sum(1 for c in deck if c.card_type == CardType.POWER)
    if card.card_type == CardType.POWER and power_count < 3:
        score += 10

    # AoE bonus: critical for multi-enemy encounters (our #1 killer)
    if card.target == TargetType.ALL_ENEMIES:
        aoe_count = sum(1 for c in deck if c.target == TargetType.ALL_ENEMIES)
        if aoe_count < 2:
            score += 15

    # Draw bonus: deck cycling is very strong
    if card.cards_draw > 0:
        score += card.cards_draw * 5

    # Multi-hit bonus: scales with Strength
    if card.hit_count > 1:
        score += 5

    # Duplicate penalty: don't pick a card we already have 2+ copies of
    copies = sum(1 for c in deck if c.id == card.id)
    if copies >= 2:
        score -= 25
    elif copies >= 1:
        score -= 10

    return score


_SIM_ML_LOADED: bool = False


def _smart_pick_or_fallback(
    offered: list[Card], deck: list[Card],
    floor: int = 1, hp: int = 50, max_hp: int = 80,
) -> Card | None:
    """Use the organic card picker (rule-based + alpha-blended ML).

    Falls back to the old tier-list picker if the new system fails.
    Lazily loads the XGBoost model on first call (not at data-load time,
    to avoid xgboost/torch conflicts in the training process).
    """
    global _SIM_ML_LOADED
    if not _SIM_ML_LOADED:
        _SIM_ML_LOADED = True
        try:
            from .card_picker import load_ml_model
            if load_ml_model():
                print("[Simulator] XGBoost card picker model loaded", flush=True)
        except Exception:
            pass
    try:
        from .card_picker import pick_card
        return pick_card(offered, deck, floor, hp, max_hp)
    except Exception:
        pass
    return _pick_card_reward(offered, deck)


def _pick_card_reward(offered: list[Card], deck: list[Card]) -> Card | None:
    """Pick the best card from offered rewards, or skip if nothing good.

    Returns None to skip. Skipping is correct when all offered cards
    would dilute the deck without adding meaningful value.
    """
    if not offered:
        return None

    scored = [(card, _score_card_for_pick(card, deck)) for card in offered]
    scored.sort(key=lambda x: x[1], reverse=True)

    best_card, best_score = scored[0]

    # Skip threshold: only pick cards that are meaningfully good
    # S-tier (100) and A-tier (80) always picked
    # B-tier (60) picked if deck is small, skipped if bloated
    # Unknown (0) never picked
    deck_size = len(deck)
    if deck_size < STRATEGY["deck_lean_target"]:
        skip_threshold = 50   # Pick B-tier and above
    else:
        skip_threshold = 65   # Only A-tier and above once deck is full

    if best_score < skip_threshold:
        return None

    return best_card


# ---------------------------------------------------------------------------
# Act 1 map model
# ---------------------------------------------------------------------------

# Act 1 (Overgrowth) has 17 rooms. Derived from real game logs:
# - Floors 1-3: weak encounters
# - Floors 4-9: normal encounters, events, shops (mid-act)
# - Floor 10: rest site (mid-act)
# - Floors 11-14: normal/elite encounters, events
# - Floor 15: event or shop
# - Floor 16: rest site (pre-boss)
# - Floor 17: boss

ROOM_TYPE = str  # "weak", "normal", "elite", "rest", "event", "boss", "shop"


def _generate_act1_map(rng: random.Random) -> list[ROOM_TYPE]:
    """Generate a sequence of rooms for Act 1.

    Based on real game logs: 17 rooms total, boss on floor 17.
    Simulates path choice by varying encounter types — the real game has
    branching paths where players can dodge hard encounters.
    """
    rooms: list[ROOM_TYPE] = []

    # Floor 1-3: weak encounters (easy early game)
    rooms.append("weak")
    rooms.append("weak")
    rooms.append("weak")

    # Floor 4-9: mix of normal, event, shop (mid-act)
    mid_rooms = ["normal", "normal", "normal", "event", "event", "shop"]
    rng.shuffle(mid_rooms)
    rooms.extend(mid_rooms)

    # Floor 10: rest site
    rooms.append("rest")

    # Floor 11-14: normal + elite (tougher section, +1 room vs old map)
    late_rooms = ["normal", "elite", rng.choice(["normal", "event"]),
                  rng.choice(["event", "shop"])]
    rng.shuffle(late_rooms)
    rooms.extend(late_rooms)

    # Floor 15: event or shop (breathing room before boss)
    rooms.append(rng.choice(["event", "shop"]))

    # Floor 16: rest (pre-boss)
    rooms.append("rest")

    # Floor 17: boss
    rooms.append("boss")

    return rooms


def _generate_act1_map_with_choices(rng: random.Random) -> list:
    """Generate Act 1 map with player-facing choices at some floors.

    Based on real game logs: 17 rooms total, boss on floor 17.
    Returns a list where each entry is either a single room type string
    (forced) or a list of 2-3 room type strings (player chooses).
    """
    rooms: list = []

    # Floor 1-3: forced weak
    rooms.extend(["weak", "weak", "weak"])

    # Floor 4-9: each offers 2-3 choices from the mid-act pool
    mid_pool = ["normal", "event", "shop", "elite"]
    for _ in range(6):
        k = rng.choice([2, 3])
        rooms.append(rng.sample(mid_pool, k=k))

    # Floor 10: forced rest
    rooms.append("rest")

    # Floor 11-14: harder choices (+1 room vs old map)
    late_pool = ["normal", "elite", "event", "rest"]
    for _ in range(4):
        rooms.append(rng.sample(late_pool, k=2))

    # Floor 15: event or shop
    rooms.append(rng.sample(["event", "shop"], k=2))

    # Floor 16: forced rest (pre-boss)
    rooms.append("rest")

    # Floor 17: forced boss
    rooms.append("boss")

    return rooms


# ---------------------------------------------------------------------------
# Dynamic room choice — mirrors the live advisor's decide_map() logic
# ---------------------------------------------------------------------------

def _choose_room(
    options: list[str],
    hp: int,
    max_hp: int,
    gold: int,
    deck_size: int,
    character: str = "SILENT",
) -> str:
    """Choose the best room from a list of options based on game state.

    Ports the live advisor's HP-threshold routing into the simulator so
    that training games make the same pathing decisions as live play.

    Priority bands:
      - HP < 35%: rest > shop > event > anything (survival mode)
      - HP < 55%: rest > shop > event > treasure > monster (cautious)
      - HP >= 55%: elite > treasure > monster > event > shop > rest (greedy)

    Gold and deck-size bonuses push toward shops when they'd be useful.
    """
    hp_pct = hp / max(1, max_hp)

    def _score(room: str) -> float:
        if room == "boss":
            return 100.0

        if hp_pct < 0.35:
            # Critical HP: survival mode
            scores = {"rest": 90, "shop": 80, "event": 60, "treasure": 50,
                      "normal": 10, "weak": 10, "elite": 0}
            return scores.get(room, 30)

        if hp_pct < 0.55:
            # Low HP: avoid elites, prefer safe nodes
            scores = {"rest": 85, "shop": 80, "event": 65, "treasure": 70,
                      "normal": 40, "weak": 40, "elite": 15}
            return scores.get(room, 30)

        # Healthy: be greedy
        scores = {"elite": 80, "normal": 55, "weak": 45, "event": 50,
                  "shop": 45, "treasure": 70, "rest": 30}
        s = scores.get(room, 40)

        # Elite bonus when HP is high
        if room == "elite" and hp_pct > 0.75:
            s += 15

        # Shop bonus when deck is large (removal value) or gold is high
        if room == "shop":
            if deck_size > 10:
                s += 15
            if gold >= 150:
                s += 25

        # Rest penalty when HP is high (don't waste it)
        if room == "rest" and hp_pct > 0.70:
            s -= 10

        # Silent-specific: push rest when HP < 50%
        if character.upper() == "SILENT" and hp_pct < 0.50 and room == "rest":
            s += 30

        return s

    return max(options, key=_score)


# ---------------------------------------------------------------------------
# Encounter selection
# ---------------------------------------------------------------------------

def _pick_encounter(
    act_data: dict,
    room_type: ROOM_TYPE,
    rng: random.Random,
    seen: set[str],
) -> str | None:
    """Pick a random encounter ID for the given room type."""
    _ensure_data_loaded()
    encounter_ids = act_data.get("encounters", [])

    candidates = []
    for eid in encounter_ids:
        enc = _ENCOUNTERS_BY_ID.get(eid, {})
        is_weak = enc.get("is_weak", False)
        room = enc.get("room_type", "Monster")

        if room_type == "weak" and is_weak:
            candidates.append(eid)
        elif room_type == "normal" and not is_weak and room == "Monster":
            candidates.append(eid)
        elif room_type == "elite" and room == "Elite":
            candidates.append(eid)
        elif room_type == "boss" and room == "Boss":
            candidates.append(eid)

    # Prefer unseen encounters
    unseen = [c for c in candidates if c not in seen]
    if unseen:
        pick = rng.choice(unseen)
    elif candidates:
        pick = rng.choice(candidates)
    else:
        return None

    seen.add(pick)
    return pick


# ---------------------------------------------------------------------------
# Potions
# ---------------------------------------------------------------------------

POTION_SLOTS = 3
POTION_DROP_CHANCE = 0.40  # 40% chance to get a potion after combat

# Simplified potion types and their effects
POTION_TYPES = [
    {"name": "Blood Potion", "heal": 20},
    {"name": "Block Potion", "block": 12},
    {"name": "Strength Potion", "strength": 2},
    {"name": "Fire Potion", "damage_all": 20},
    {"name": "Weak Potion", "enemy_weak": 3},
]


# ---------------------------------------------------------------------------
# Combat simulation
# ---------------------------------------------------------------------------

MAX_COMBAT_TURNS = 30  # Safety cap


@dataclass
class CombatResult:
    outcome: str  # "win" or "lose"
    turns: int
    hp_before: int
    hp_after: int
    encounter_id: str
    gold_earned: int = 0


def simulate_combat(
    deck: list[Card],
    player_hp: int,
    player_max_hp: int,
    player_max_energy: int,
    encounter_id: str,
    card_db: CardDB,
    rng: random.Random,
    potions: list[dict] | None = None,
    solver_time_limit_ms: float = 500.0,
    is_boss: bool = False,
) -> tuple[CombatResult, list[dict]]:
    """Run a full combat from start to finish using the solver.

    Returns (CombatResult, remaining_potions).
    """
    _ensure_data_loaded()
    enc = _ENCOUNTERS_BY_ID.get(encounter_id, {})
    monster_list = enc.get("monsters", [])
    potions = list(potions) if potions else []

    # Spawn enemies
    enemies: list[EnemyState] = []
    enemy_ais: list[EnemyAI] = []
    for m in monster_list:
        mid = m["id"]
        enemy = _spawn_enemy(mid)
        enemies.append(enemy)
        enemy_ais.append(_create_enemy_ai(mid))

    if not enemies:
        return CombatResult("win", 0, player_hp, player_hp, encounter_id), potions

    # Build player state
    draw_pile = list(deck)
    rng.shuffle(draw_pile)

    player = PlayerState(
        hp=player_hp,
        max_hp=player_max_hp,
        energy=player_max_energy,
        max_energy=player_max_energy,
        draw_pile=draw_pile,
    )

    state = CombatState(player=player, enemies=enemies)

    hp_before = player_hp

    # Boss fights: dump ALL offensive potions immediately + heal at 40% HP.
    # Non-boss: save offensive potions for the boss, only emergency heal.
    if is_boss:
        potions = _use_precombat_potions(state, potions)
        # Also use healing potion if HP is below 40% (HP resets next act,
        # so surviving is all that matters)
        if state.player.hp < state.player.max_hp * 0.40:
            potions = _use_emergency_potion(state, potions)

    # Set initial enemy intents
    _set_enemy_intents(state, enemy_ais)

    for turn_num in range(1, MAX_COMBAT_TURNS + 1):
        # Start player turn
        start_turn(state)

        # Check combat over (enemy might have died from start-of-turn effects)
        result = is_combat_over(state)
        if result:
            return CombatResult(
                result, turn_num, hp_before,
                max(0, state.player.hp), encounter_id,
            ), potions

        # Emergency potion use: heal if HP critically low
        # Boss: lower threshold (35%) since we already used healing pre-combat
        # Non-boss: standard 25% threshold
        emergency_threshold = 0.35 if is_boss else 0.25
        if state.player.hp < state.player.max_hp * emergency_threshold:
            potions = _use_emergency_potion(state, potions)

        # Solve: find best card play sequence
        solve_result = solve_turn(
            state, card_db=card_db,
            time_limit_ms=solver_time_limit_ms,
        )

        # Execute the solver's chosen actions
        for action in solve_result.actions:
            if action.action_type == "end_turn":
                break
            if action.card_idx is not None:
                try:
                    play_card(state, action.card_idx,
                              target_idx=action.target_idx, card_db=card_db)
                except (IndexError, ValueError):
                    break

            result = is_combat_over(state)
            if result:
                return CombatResult(
                    result, turn_num, hp_before,
                    max(0, state.player.hp), encounter_id,
                ), potions

        # End player turn
        end_turn(state)

        # Resolve enemy intents (damage to player)
        resolve_enemy_intents(state)
        # Apply buff/debuff effects from the move tables
        _resolve_sim_intents(state, enemy_ais)
        # Tick enemy debuffs/poison AFTER intents resolve
        tick_enemy_powers(state)

        result = is_combat_over(state)
        if result:
            return CombatResult(
                result, turn_num, hp_before,
                max(0, state.player.hp), encounter_id,
            ), potions

        # Set new enemy intents for next turn
        _set_enemy_intents(state, enemy_ais)

    # Ran out of turns — treat as loss
    return CombatResult("lose", MAX_COMBAT_TURNS, hp_before,
                        max(0, state.player.hp), encounter_id), potions


def _use_precombat_potions(
    state: CombatState, potions: list[dict],
) -> list[dict]:
    """Use offensive potions at combat start (Strength, Fire, Weak)."""
    remaining = []
    for pot in potions:
        used = False
        if pot.get("strength"):
            state.player.powers["Strength"] = (
                state.player.powers.get("Strength", 0) + pot["strength"]
            )
            used = True
        elif pot.get("damage_all"):
            for e in state.enemies:
                if e.is_alive:
                    e.hp -= pot["damage_all"]
            used = True
        elif pot.get("enemy_weak"):
            for e in state.enemies:
                if e.is_alive:
                    e.powers["Weak"] = e.powers.get("Weak", 0) + pot["enemy_weak"]
            used = True
        if not used:
            remaining.append(pot)
    return remaining


def _use_emergency_potion(
    state: CombatState, potions: list[dict],
) -> list[dict]:
    """Use a healing potion if available."""
    remaining = []
    healed = False
    for pot in potions:
        if pot.get("heal") and not healed:
            state.player.hp = min(
                state.player.hp + pot["heal"], state.player.max_hp
            )
            healed = True
        else:
            remaining.append(pot)
    return remaining


def _set_enemy_intents(state: CombatState, ais: list[EnemyAI]) -> None:
    """Set intents on all living enemies using their AI.

    Stores the full intent (including buff/debuff data) on the AI so
    _resolve_sim_intents() can apply them after the player's turn.
    """
    for enemy, ai in zip(state.enemies, ais):
        if not enemy.is_alive:
            continue
        intent = ai.pick_intent()
        enemy.intent_type = intent.get("type", "Attack")
        enemy.intent_damage = intent.get("damage")
        enemy.intent_hits = intent.get("hits", 1)
        enemy.intent_block = intent.get("block")
        # Stash full intent for post-turn resolution
        ai._pending_intent = intent


def _resolve_sim_intents(state: CombatState, ais: list[EnemyAI]) -> None:
    """Resolve buff/debuff effects from enemy intents.

    Called AFTER resolve_enemy_intents() (which handles Attack/Defend).
    This applies the mechanical effects that the base engine doesn't know about.
    """
    for enemy, ai in zip(state.enemies, ais):
        if not enemy.is_alive:
            continue
        intent = getattr(ai, '_pending_intent', None)
        if not intent:
            continue

        # Self-buffs
        if intent.get("self_strength"):
            enemy.powers["Strength"] = (
                enemy.powers.get("Strength", 0) + intent["self_strength"]
            )
        if intent.get("self_block"):
            enemy.block += intent["self_block"]

        # All-ally buffs (like Brute Roar, Kin Priest Ritual)
        if intent.get("all_strength"):
            for e in state.enemies:
                if e.is_alive:
                    e.powers["Strength"] = (
                        e.powers.get("Strength", 0) + intent["all_strength"]
                    )

        # Player debuffs
        if intent.get("player_weak"):
            state.player.powers["Weak"] = (
                state.player.powers.get("Weak", 0) + intent["player_weak"]
            )
        if intent.get("player_frail"):
            state.player.powers["Frail"] = (
                state.player.powers.get("Frail", 0) + intent["player_frail"]
            )
        if intent.get("player_vulnerable"):
            state.player.powers["Vulnerable"] = (
                state.player.powers.get("Vulnerable", 0)
                + intent["player_vulnerable"]
            )
        if intent.get("player_shrink"):
            state.player.powers["Shrink"] = (
                state.player.powers.get("Shrink", 0)
                - intent["player_shrink"]  # Shrink is stored as negative value
            )

        if intent.get("player_constrict"):
            state.player.powers["Constrict"] = (
                state.player.powers.get("Constrict", 0)
                + intent["player_constrict"]
            )
        if intent.get("player_tangled"):
            state.player.powers["Tangled"] = (
                state.player.powers.get("Tangled", 0)
                + intent["player_tangled"]
            )

        ai._pending_intent = None


# ---------------------------------------------------------------------------
# Event simulation
# ---------------------------------------------------------------------------

def _simulate_event(
    event_id: str,
    deck: list[Card],
    hp: int,
    max_hp: int,
    gold: int,
    card_db: CardDB,
    rng: random.Random,
) -> dict:
    """Simulate an event and return state changes.

    Returns dict with keys: hp_delta, max_hp_delta, gold_delta,
    cards_added, cards_removed.
    """
    _ensure_data_loaded()
    event = _EVENTS_BY_ID.get(event_id)
    if not event:
        return {"hp_delta": 0, "max_hp_delta": 0, "gold_delta": 0,
                "cards_added": [], "cards_removed": []}

    options = event.get("options", [])
    if not options:
        return {"hp_delta": 0, "max_hp_delta": 0, "gold_delta": 0,
                "cards_added": [], "cards_removed": []}

    # Simple heuristic: parse option descriptions for effects
    best_option = _evaluate_event_options(options, hp, max_hp, gold, deck)
    return _apply_event_option(best_option, hp, max_hp, deck, card_db, rng)


def _evaluate_event_options(
    options: list[dict],
    hp: int, max_hp: int, gold: int,
    deck: list[Card],
) -> dict:
    """Pick the best event option using simple heuristics."""
    hp_pct = hp / max_hp if max_hp > 0 else 1.0

    best_score = float("-inf")
    best_option = options[0] if options else {}

    for opt in options:
        desc = (opt.get("description") or "").lower()
        score = 0.0

        # Positive effects
        if "heal" in desc:
            score += 20 * (1.0 - hp_pct)  # Healing more valuable when low
        if "max hp" in desc and "gain" in desc:
            score += 15
        if "upgrade" in desc:
            score += 12
        if "transform" in desc:
            score += 8
        if "remove" in desc and "card" in desc:
            score += 15  # Card removal is very valuable
        if "gold" in desc and "gain" in desc:
            score += 5
        if "relic" in desc:
            score += 20

        # Negative effects
        if "damage" in desc or "lose" in desc:
            if hp_pct < 0.4:
                score -= 30  # Too dangerous when low
            else:
                score -= 8
        if "curse" in desc:
            score -= 20

        if score > best_score:
            best_score = score
            best_option = opt

    return best_option


def _apply_event_option(
    option: dict,
    hp: int, max_hp: int,
    deck: list[Card],
    card_db: CardDB,
    rng: random.Random,
) -> dict:
    """Apply an event option and return changes.

    Since we can't perfectly parse all event descriptions, we approximate
    common patterns.
    """
    import re
    desc = (option.get("description") or "").lower()
    result = {"hp_delta": 0, "max_hp_delta": 0, "gold_delta": 0,
              "cards_added": [], "cards_removed": []}

    # Heal N HP
    heal_match = re.search(r'heal\s*(\d+)', desc)
    if heal_match:
        result["hp_delta"] = int(heal_match.group(1))

    # Gain N Max HP
    max_hp_match = re.search(r'gain\s*(\d+)\s*max hp', desc)
    if max_hp_match:
        result["max_hp_delta"] = int(max_hp_match.group(1))

    # Take N damage / Lose N HP
    dmg_match = re.search(r'(?:take|lose)\s*(\d+)\s*(?:damage|hp)', desc)
    if dmg_match:
        result["hp_delta"] -= int(dmg_match.group(1))

    # Gain N gold
    gold_match = re.search(r'gain\s*(\d+)\s*gold', desc)
    if gold_match:
        result["gold_delta"] = int(gold_match.group(1))

    # Lose N gold
    gold_lose_match = re.search(r'lose\s*(\d+)\s*gold', desc)
    if gold_lose_match:
        result["gold_delta"] -= int(gold_lose_match.group(1))

    return result


# ---------------------------------------------------------------------------
# Rest site logic
# ---------------------------------------------------------------------------

def _rest_site_decision(
    hp: int, max_hp: int,
    deck: list[Card],
    card_db: CardDB,
    rng: random.Random,
    character: str = "IRONCLAD",
    floor: int = 10,
) -> dict:
    """Decide rest site action: rest (heal) or smith (upgrade).

    Character-specific HP thresholds (matching the live advisor):
      - Silent: rest below 50%, always upgrade above 70%
      - Ironclad: rest below 40%, always upgrade above 60%
    Gray zone: upgrade if the deck has a high-value unupgraded card.

    Also uses the organic card picker scoring to identify the best
    upgrade target by intrinsic card power.

    Returns dict with hp_delta and optionally upgraded card index.
    """
    hp_pct = hp / max_hp if max_hp > 0 else 1.0

    # Character-specific thresholds
    is_silent = character.upper() == "SILENT"
    rest_threshold = 0.50 if is_silent else 0.40
    upgrade_threshold = 0.70 if is_silent else 0.60

    # Always rest when HP is critical
    if hp_pct < rest_threshold:
        heal = int(max_hp * 0.3)
        return {"action": "rest", "hp_delta": heal, "upgrade_card_idx": None}

    # Score upgradeable cards using the organic picker's power scoring
    upgradeable = []
    for i, card in enumerate(deck):
        if card.upgraded:
            continue
        if card.card_type in (CardType.STATUS, CardType.CURSE):
            continue
        # Skip basic cards — upgrading Strike/Defend is almost never worth it
        if card.name in ("Strike", "Defend"):
            continue

        score = 0.0
        try:
            from .card_picker import extract_properties, _card_power_score
            props = extract_properties(card)
            score = _card_power_score(card, props) * 100

            # Bonus for upgrading cards with large stat deltas
            upgraded = card_db.get_upgraded(card.id)
            if upgraded:
                uprops = extract_properties(upgraded)
                # Damage improvement
                if uprops.deals_damage > props.deals_damage:
                    score += (uprops.deals_damage - props.deals_damage) * 2
                # Block improvement
                if uprops.grants_block > props.grants_block:
                    score += (uprops.grants_block - props.grants_block) * 2
                # Draw improvement
                if uprops.draws_cards > props.draws_cards:
                    score += (uprops.draws_cards - props.draws_cards) * 10
                # Poison improvement
                if uprops.applies_poison > props.applies_poison:
                    score += (uprops.applies_poison - props.applies_poison) * 3

            # Powers are high-priority upgrades (permanent effects)
            if props.is_power:
                score += 15
        except Exception:
            # Fallback: tier-based scoring
            _init_tier_scores()
            score = _TIER_SCORES.get(card.name.lower(), 40)
            upgraded = card_db.get_upgraded(card.id)
            if upgraded:
                if upgraded.damage and card.damage:
                    score += (upgraded.damage - card.damage) * 2
                if upgraded.block and card.block:
                    score += (upgraded.block - card.block) * 2

        upgradeable.append((i, score))

    if not upgradeable:
        # Nothing to upgrade, rest instead
        heal = int(max_hp * 0.3)
        return {"action": "rest", "hp_delta": heal, "upgrade_card_idx": None}

    upgradeable.sort(key=lambda x: x[1], reverse=True)
    best_idx, best_score = upgradeable[0]

    # Always upgrade when HP is high
    if hp_pct >= upgrade_threshold:
        return {"action": "smith", "hp_delta": 0, "upgrade_card_idx": best_idx}

    # Gray zone: upgrade if we have a genuinely high-value target
    if best_score >= 50:
        return {"action": "smith", "hp_delta": 0, "upgrade_card_idx": best_idx}

    # Otherwise rest
    heal = int(max_hp * 0.3)
    return {"action": "rest", "hp_delta": heal, "upgrade_card_idx": None}


# ---------------------------------------------------------------------------
# Shop simulation — archetype-aware, multi-step
# ---------------------------------------------------------------------------

SHOP_CARD_REMOVE_COST = 75
SHOP_CARD_COSTS = {"Common": 50, "Uncommon": 75, "Rare": 150}
SHOP_POTION_COST = 50  # Flat cost for any potion
SHOP_RELIC_COST = 150   # Simplified flat relic price

# Cards that should never be removed — core identity cards
_PROTECTED_CARDS = frozenset({"Survivor", "Neutralize", "Bash", "Eruption", "Vigilance"})


def _score_card_for_removal(card: Card, deck: list[Card], floor: int,
                            hp: int, max_hp: int) -> float:
    """Score a card for removal — lower score = better removal candidate.

    Uses intrinsic card power + archetype alignment but weights differently
    from the *pick* scorer: a card already in the deck has proven value
    from its raw stats (damage, block, draw) even if off-archetype.
    The alignment penalty is halved so useful generics (Dash, Backflip)
    aren't removal targets just because they don't match the archetype.

    Protected cards get a high score so they are never removed.
    """
    if card.name in _PROTECTED_CARDS:
        return 999.0

    try:
        from .card_picker import (extract_properties, build_signature,
                                  _card_power_score, _alignment_score)

        props = extract_properties(card)
        sig = build_signature(deck)

        # Intrinsic power is the main driver for removal scoring
        power = _card_power_score(card, props)
        # Halve the alignment penalty — off-archetype cards that are
        # already in the deck still contribute raw stats
        alignment = _alignment_score(card, props, sig) * 0.5

        # Upgraded cards are more valuable than their base versions
        upgrade_bonus = 0.05 if card.upgraded else 0.0

        return max(0.01, power + alignment + upgrade_bonus)
    except Exception:
        # Fallback: basic starters are worst, everything else is neutral
        if card.name in ("Strike", "Defend") and not card.upgraded:
            return 0.05
        return 0.50


def _score_relic_for_purchase(relic_name: str, deck: list[Card],
                              character: str) -> float:
    """Score a relic for purchase based on archetype fit.

    Returns 0.0–2.0.  Uses the RELIC_GUIDE from config if available.
    """
    try:
        from .config_a import RELIC_GUIDE
        from .card_picker import build_signature

        sig = build_signature(deck)
        archetype = sig.dominant_archetype

        guide = RELIC_GUIDE.get(character, {})

        # Check avoid list first
        avoid = guide.get("avoid", {}).get("relics", [])
        if relic_name in avoid:
            return -1.0

        # Top picks are always good
        top = guide.get("top_picks", {}).get("relics", [])
        if relic_name in top:
            return 2.0

        # Archetype-specific match
        archetype_to_category = {
            "poison": "poison_synergy",
            "shiv": "shiv_synergy",
            "sly": "sly_synergy",
        }
        if archetype in archetype_to_category:
            cat_key = archetype_to_category[archetype]
            cat_relics = guide.get(cat_key, {}).get("relics", [])
            if relic_name in cat_relics:
                return 1.5

        # Partial match — in some category but not our archetype
        for key, info in guide.items():
            if key in ("top_picks", "avoid"):
                continue
            if relic_name in info.get("relics", []):
                return 0.5

        return 0.0
    except Exception:
        return 0.0


def _simulate_shop(
    deck: list[Card],
    gold: int,
    card_db: CardDB,
    pools: dict[str, list[Card]],
    rng: random.Random,
    floor: int = 8,
    hp: int = 50,
    max_hp: int = 80,
    character: str = "SILENT",
    potions: list[dict] | None = None,
) -> dict:
    """Simulate a shop visit with archetype-aware multi-step purchasing.

    Priority order:
      1. Remove the weakest card (by organic scorer, not just Strike/Defend)
      2. Buy a relic that fits the archetype (if score >= 1.0)
      3. Buy an archetype-fitting card (using score_card)
      4. Buy a potion if HP is low and we have room

    Returns: {gold_delta, cards_added, cards_removed, card_upgraded,
              relics_bought, potions_bought}
    """
    if potions is None:
        potions = []

    result = {
        "gold_delta": 0,
        "cards_added": [],
        "cards_removed": [],
        "card_upgraded": None,
        "relics_bought": [],
        "potions_bought": [],
    }

    # --- Step 1: Remove the weakest card in the deck ---
    if gold >= SHOP_CARD_REMOVE_COST and len(deck) >= 8:
        # Score every card; find the worst contributor
        scored = [
            (i, c, _score_card_for_removal(c, deck, floor, hp, max_hp))
            for i, c in enumerate(deck)
        ]
        scored.sort(key=lambda x: x[2])  # Lowest score = worst card

        worst_idx, worst_card, worst_score = scored[0]
        # Only remove if the card is genuinely weak (below 0.25 contribution)
        # or is an unupgraded basic (Strike/Defend always worth removing)
        is_basic = worst_card.name in ("Strike", "Defend") and not worst_card.upgraded
        if worst_score < 0.25 or is_basic:
            result["cards_removed"].append(worst_idx)
            result["gold_delta"] -= SHOP_CARD_REMOVE_COST
            gold -= SHOP_CARD_REMOVE_COST

    # --- Step 2: Buy a relic (if affordable and good fit) ---
    if gold >= SHOP_RELIC_COST:
        # Simulate shop relic offering: pick 3 from available relics
        _ensure_data_loaded()
        all_relic_names = [r.get("name", r["id"]) for r in _RELICS_BY_ID.values()]
        if all_relic_names:
            shop_relics = rng.sample(all_relic_names,
                                     min(3, len(all_relic_names)))
            best_relic, best_relic_score = None, 0.0
            for rname in shop_relics:
                score = _score_relic_for_purchase(rname, deck, character)
                if score > best_relic_score:
                    best_relic = rname
                    best_relic_score = score

            # Only buy relics that are top picks or strong archetype matches
            if best_relic_score >= 1.0 and best_relic is not None:
                result["relics_bought"].append(best_relic)
                result["gold_delta"] -= SHOP_RELIC_COST
                gold -= SHOP_RELIC_COST

    # --- Step 3: Buy a card (archetype-aware, using organic scorer) ---
    if gold >= 50:
        # Shop offers more cards than combat rewards (typically 6-7 in STS2)
        offered = _offer_card_rewards(pools, deck, 6)
        # Use the organic picker for scoring
        pick = _smart_pick_or_fallback(offered, deck, floor, hp, max_hp)
        if pick:
            # Determine cost by rarity
            cost = 75  # Default (Uncommon)
            for rarity, cards in pools.items():
                if any(c.id == pick.id for c in cards):
                    cost = SHOP_CARD_COSTS.get(rarity, 75)
                    break
            if gold >= cost:
                result["cards_added"].append(pick)
                result["gold_delta"] -= cost
                gold -= cost

    # --- Step 4: Buy a potion if HP is low and we have room ---
    hp_ratio = hp / max(1, max_hp)
    potion_count = len(potions) + len(result["potions_bought"])
    if (gold >= SHOP_POTION_COST
            and hp_ratio < 0.55
            and potion_count < POTION_SLOTS):
        # Pick a useful potion — heal potion prioritised when low HP
        if hp_ratio < 0.35:
            pot = {"name": "Blood Potion", "heal": 20}
        else:
            pot = rng.choice(POTION_TYPES)
        result["potions_bought"].append(pot)
        result["gold_delta"] -= SHOP_POTION_COST
        gold -= SHOP_POTION_COST

    return result


# ---------------------------------------------------------------------------
# Gold rewards
# ---------------------------------------------------------------------------

GOLD_REWARDS = {
    "weak": (10, 20),
    "normal": (15, 25),
    "elite": (25, 40),
    "boss": (50, 100),
}


# ---------------------------------------------------------------------------
# Full Act 1 simulation
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Result of a complete Act 1 simulation."""
    run_id: int
    outcome: str           # "win" (beat boss), "lose" (died)
    floor_reached: int     # Last floor completed
    final_hp: int
    max_hp: int
    gold: int
    deck_size: int
    combats_won: int
    combats_fought: int
    total_turns: int
    death_encounter: str | None = None
    cards_picked: list[str] = field(default_factory=list)
    cards_skipped: int = 0
    events_visited: int = 0
    rests_taken: int = 0
    upgrades_done: int = 0
    elapsed_ms: float = 0.0
    combat_log: list[dict] = field(default_factory=list)


def simulate_act1(
    run_id: int = 0,
    character: str = "IRONCLAD",
    seed: int | None = None,
    solver_time_limit_ms: float = 200.0,
    verbose: bool = False,
) -> RunResult:
    """Simulate a complete Act 1 (Overgrowth) run.

    Args:
        run_id: Identifier for this run.
        character: Character ID (e.g., "IRONCLAD").
        seed: Random seed for reproducibility.
        solver_time_limit_ms: Time limit per combat turn solve.
        verbose: Print progress.

    Returns:
        RunResult with full statistics.
    """
    t0 = time.perf_counter()
    rng = random.Random(seed)
    # Seed the global random too (used by combat engine shuffle)
    random.seed(seed)

    _ensure_data_loaded()
    card_db = load_cards()

    # Character setup
    char_data = _CHARACTERS_BY_ID.get(character, {})
    hp = char_data.get("starting_hp", 80)
    max_hp = hp
    gold = char_data.get("starting_gold", 99)
    max_energy = char_data.get("max_energy", 3)

    # Build starting deck
    raw_deck_ids = char_data.get("starting_deck", [])
    deck: list[Card] = []
    for raw_id in raw_deck_ids:
        card_id = _normalize_card_id(raw_id)
        card = card_db.get(card_id)
        if card:
            deck.append(card)
        else:
            # Try direct lookup
            card = card_db.get(raw_id)
            if card:
                deck.append(card)

    if not deck:
        # Fallback: hardcode Ironclad starter
        for _ in range(5):
            c = card_db.get("STRIKE_IRONCLAD")
            if c:
                deck.append(c)
        for _ in range(4):
            c = card_db.get("DEFEND_IRONCLAD")
            if c:
                deck.append(c)
        c = card_db.get("BASH")
        if c:
            deck.append(c)

    # Card pools for rewards
    char_color = char_data.get("color", "red")
    # Map character color names to card color field
    color_map = {"red": "ironclad", "green": "silent", "blue": "defect",
                 "purple": "necrobinder", "yellow": "regent"}
    card_color = color_map.get(char_color, char_color)
    pools = _build_card_pool(card_db, card_color)

    # Act data
    act_data = _ACTS_BY_ID.get("OVERGROWTH", {})

    # Generate map with choices (dynamic pathing based on game state)
    map_with_choices = _generate_act1_map_with_choices(rng)

    # Potions
    potions: list[dict] = []  # Start with no potions (acquired from combat)

    # Run state
    result = RunResult(run_id=run_id, outcome="lose", floor_reached=0,
                       final_hp=hp, max_hp=max_hp, gold=gold,
                       deck_size=len(deck), combats_won=0, combats_fought=0,
                       total_turns=0)

    seen_encounters: set[str] = set()
    events_list = list(act_data.get("events", []))
    rng.shuffle(events_list)
    event_idx = 0

    for floor_num, floor_entry in enumerate(map_with_choices, 1):
        result.floor_reached = floor_num

        # Resolve room type: either forced (string) or chosen from options (list)
        if isinstance(floor_entry, list):
            room_type = _choose_room(
                floor_entry, hp, max_hp, gold, len(deck), character)
        else:
            room_type = floor_entry

        if verbose:
            choices_str = ""
            if isinstance(floor_entry, list):
                choices_str = f" (chose from {floor_entry})"
            print(f"  Floor {floor_num}: {room_type}{choices_str} "
                  f"(HP: {hp}/{max_hp}, Gold: {gold}, Deck: {len(deck)})")

        if room_type in ("weak", "normal", "elite", "boss"):
            # Pick encounter
            enc_id = _pick_encounter(act_data, room_type, rng, seen_encounters)
            if enc_id is None:
                continue

            # Run combat (boss fights get is_boss=True for potion dump + deeper search)
            _is_boss = (room_type == "boss")
            combat, potions = simulate_combat(
                deck=deck, player_hp=hp, player_max_hp=max_hp,
                player_max_energy=max_energy, encounter_id=enc_id,
                card_db=card_db, rng=rng, potions=potions,
                solver_time_limit_ms=solver_time_limit_ms,
                is_boss=_is_boss,
            )
            result.combats_fought += 1
            result.total_turns += combat.turns

            result.combat_log.append({
                "floor": floor_num,
                "encounter": enc_id,
                "room_type": room_type,
                "outcome": combat.outcome,
                "turns": combat.turns,
                "hp_before": combat.hp_before,
                "hp_after": combat.hp_after,
            })

            if verbose:
                print(f"    Combat: {enc_id} -> {combat.outcome} "
                      f"({combat.turns}T, HP: {combat.hp_before}->{combat.hp_after})")

            if combat.outcome == "lose":
                result.outcome = "lose"
                result.death_encounter = enc_id
                result.final_hp = 0
                break

            result.combats_won += 1
            hp = combat.hp_after

            # Gold reward
            gold_range = GOLD_REWARDS.get(room_type, (10, 20))
            gold_earned = rng.randint(*gold_range)
            gold += gold_earned

            # Burning Blood relic: heal 6 HP after combat (Ironclad)
            if character == "IRONCLAD":
                hp = min(hp + 6, max_hp)

            # Potion drop
            if rng.random() < POTION_DROP_CHANCE and len(potions) < POTION_SLOTS:
                pot = rng.choice(POTION_TYPES)
                potions.append(dict(pot))

            # Card reward (not for boss — boss gives relic)
            if room_type != "boss":
                offered = _offer_card_rewards(pools, deck)
                # Use organic picker (rule-based + alpha-blended ML)
                pick = _smart_pick_or_fallback(offered, deck, floor_num, hp, max_hp)
                if pick:
                    deck.append(pick)
                    result.cards_picked.append(pick.name)
                    if verbose:
                        print(f"    Picked: {pick.name}")
                else:
                    result.cards_skipped += 1

            if room_type == "boss":
                result.outcome = "win"

        elif room_type == "rest":
            decision = _rest_site_decision(hp, max_hp, deck, card_db, rng,
                                           character=character, floor=floor_num)
            if decision["action"] == "rest":
                hp = min(hp + decision["hp_delta"], max_hp)
                result.rests_taken += 1
                if verbose:
                    print(f"    Rest: healed to {hp}/{max_hp}")
            else:
                idx = decision["upgrade_card_idx"]
                if idx is not None and idx < len(deck):
                    upgraded = card_db.get_upgraded(deck[idx].id)
                    if upgraded:
                        old_name = deck[idx].name
                        deck[idx] = upgraded
                        result.upgrades_done += 1
                        if verbose:
                            print(f"    Smith: upgraded {old_name}")

        elif room_type == "event":
            if event_idx < len(events_list):
                eid = events_list[event_idx]
                event_idx += 1
            else:
                eid = rng.choice(events_list) if events_list else None

            if eid:
                changes = _simulate_event(eid, deck, hp, max_hp, gold,
                                          card_db, rng)
                hp = max(1, min(hp + changes["hp_delta"],
                                max_hp + changes["max_hp_delta"]))
                max_hp += changes["max_hp_delta"]
                gold = max(0, gold + changes["gold_delta"])

                # Remove cards (by index, descending to avoid shifting)
                for idx in sorted(changes["cards_removed"], reverse=True):
                    if idx < len(deck):
                        deck.pop(idx)

                # Add cards
                for card in changes["cards_added"]:
                    deck.append(card)

                result.events_visited += 1
                if verbose:
                    print(f"    Event: {eid} (HP: {hp}/{max_hp})")

        elif room_type == "shop":
            shop_result = _simulate_shop(
                deck, gold, card_db, pools, rng,
                floor=floor_num, hp=hp, max_hp=max_hp,
                character=character, potions=potions,
            )
            gold += shop_result["gold_delta"]

            # Remove cards (descending index)
            for idx in sorted(shop_result["cards_removed"], reverse=True):
                if idx < len(deck):
                    removed_name = deck[idx].name
                    deck.pop(idx)
                    if verbose:
                        print(f"    Shop removed: {removed_name}")

            for card in shop_result["cards_added"]:
                deck.append(card)
                result.cards_picked.append(card.name)
                if verbose:
                    print(f"    Shop bought card: {card.name}")

            for relic_name in shop_result.get("relics_bought", []):
                if verbose:
                    print(f"    Shop bought relic: {relic_name}")

            for pot in shop_result.get("potions_bought", []):
                potions.append(dict(pot))
                if verbose:
                    print(f"    Shop bought potion: {pot['name']}")

            if verbose:
                print(f"    Shop: gold now {gold}")

    # Finalize
    result.final_hp = hp
    result.max_hp = max_hp
    result.gold = gold
    result.deck_size = len(deck)
    result.elapsed_ms = (time.perf_counter() - t0) * 1000
    return result


# ---------------------------------------------------------------------------
# Batch runner and statistics
# ---------------------------------------------------------------------------

@dataclass
class BatchStats:
    """Aggregated statistics from many runs."""
    total_runs: int
    wins: int
    losses: int
    win_rate: float
    avg_floor: float
    median_floor: float
    avg_final_hp: float
    avg_deck_size: float
    avg_combats_won: float
    avg_turns_per_combat: float
    avg_run_time_ms: float
    total_time_s: float
    # Death encounter frequency
    death_encounters: dict[str, int]
    # Card pick frequency
    card_picks: dict[str, int]
    # Floor reached histogram
    floor_histogram: dict[int, int]
    # Per-run results for CSV export
    runs: list[RunResult]


def run_batch(
    num_runs: int = 100,
    character: str = "IRONCLAD",
    base_seed: int | None = None,
    solver_time_limit_ms: float = 200.0,
    verbose: bool = False,
    progress: bool = True,
) -> BatchStats:
    """Run multiple simulations and collect statistics."""
    t0 = time.perf_counter()
    results: list[RunResult] = []

    for i in range(num_runs):
        seed = (base_seed + i) if base_seed is not None else None
        r = simulate_act1(
            run_id=i,
            character=character,
            seed=seed,
            solver_time_limit_ms=solver_time_limit_ms,
            verbose=verbose,
        )
        results.append(r)

        if progress and (i + 1) % max(1, num_runs // 20) == 0:
            wins = sum(1 for r in results if r.outcome == "win")
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            print(f"  [{i+1}/{num_runs}] Win rate: {wins}/{i+1} "
                  f"({100*wins/(i+1):.1f}%) | {rate:.1f} runs/sec")

    # Aggregate
    wins = sum(1 for r in results if r.outcome == "win")
    floors = [r.floor_reached for r in results]
    final_hps = [r.final_hp for r in results]
    deck_sizes = [r.deck_size for r in results]
    combats_won = [r.combats_won for r in results]
    total_turns = [r.total_turns for r in results]
    combats_fought = [r.combats_fought for r in results]

    avg_turns_per = (
        sum(total_turns) / sum(combats_fought)
        if sum(combats_fought) > 0 else 0
    )

    # Death encounters
    death_enc: dict[str, int] = {}
    for r in results:
        if r.death_encounter:
            death_enc[r.death_encounter] = death_enc.get(r.death_encounter, 0) + 1

    # Card picks
    card_picks: dict[str, int] = {}
    for r in results:
        for name in r.cards_picked:
            card_picks[name] = card_picks.get(name, 0) + 1

    # Floor histogram
    floor_hist: dict[int, int] = {}
    for f in floors:
        floor_hist[f] = floor_hist.get(f, 0) + 1

    total_time = time.perf_counter() - t0

    return BatchStats(
        total_runs=num_runs,
        wins=wins,
        losses=num_runs - wins,
        win_rate=wins / num_runs if num_runs > 0 else 0,
        avg_floor=statistics.mean(floors) if floors else 0,
        median_floor=statistics.median(floors) if floors else 0,
        avg_final_hp=statistics.mean(final_hps) if final_hps else 0,
        avg_deck_size=statistics.mean(deck_sizes) if deck_sizes else 0,
        avg_combats_won=statistics.mean(combats_won) if combats_won else 0,
        avg_turns_per_combat=avg_turns_per,
        avg_run_time_ms=statistics.mean([r.elapsed_ms for r in results]),
        total_time_s=total_time,
        death_encounters=death_enc,
        card_picks=card_picks,
        floor_histogram=floor_hist,
        runs=results,
    )


def print_stats(stats: BatchStats) -> None:
    """Print a formatted summary of batch statistics."""
    print("\n" + "=" * 60)
    print(f"  ACT 1 SIMULATION RESULTS  ({stats.total_runs} runs)")
    print("=" * 60)

    print(f"\n  Win rate:              {stats.wins}/{stats.total_runs} "
          f"({100*stats.win_rate:.1f}%)")
    print(f"  Avg floor reached:     {stats.avg_floor:.1f}")
    print(f"  Median floor reached:  {stats.median_floor:.0f}")
    print(f"  Avg final HP:          {stats.avg_final_hp:.1f}")
    print(f"  Avg deck size:         {stats.avg_deck_size:.1f}")
    print(f"  Avg combats won:       {stats.avg_combats_won:.1f}")
    print(f"  Avg turns/combat:      {stats.avg_turns_per_combat:.1f}")
    print(f"  Avg run time:          {stats.avg_run_time_ms:.0f}ms")
    print(f"  Total time:            {stats.total_time_s:.1f}s")

    if stats.death_encounters:
        print(f"\n  Deaths by encounter:")
        sorted_deaths = sorted(stats.death_encounters.items(),
                               key=lambda x: x[1], reverse=True)
        for enc, count in sorted_deaths[:10]:
            pct = 100 * count / stats.losses if stats.losses > 0 else 0
            print(f"    {enc:40s} {count:4d} ({pct:.1f}%)")

    if stats.card_picks:
        print(f"\n  Most picked cards:")
        sorted_picks = sorted(stats.card_picks.items(),
                              key=lambda x: x[1], reverse=True)
        for name, count in sorted_picks[:15]:
            print(f"    {name:30s} {count:4d}")

    if stats.floor_histogram:
        print(f"\n  Floor reached histogram:")
        for floor in sorted(stats.floor_histogram.keys()):
            count = stats.floor_histogram[floor]
            bar = "#" * (count * 40 // stats.total_runs)
            print(f"    Floor {floor:2d}: {count:4d} {bar}")

    print()


def export_csv(stats: BatchStats, path: str) -> None:
    """Export per-run results to CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "run_id", "outcome", "floor_reached", "final_hp", "max_hp",
            "gold", "deck_size", "combats_won", "combats_fought",
            "total_turns", "death_encounter", "cards_picked",
            "cards_skipped", "events_visited", "rests_taken",
            "upgrades_done", "elapsed_ms",
        ])
        for r in stats.runs:
            writer.writerow([
                r.run_id, r.outcome, r.floor_reached, r.final_hp, r.max_hp,
                r.gold, r.deck_size, r.combats_won, r.combats_fought,
                r.total_turns, r.death_encounter or "",
                "|".join(r.cards_picked), r.cards_skipped,
                r.events_visited, r.rests_taken, r.upgrades_done,
                f"{r.elapsed_ms:.1f}",
            ])
    print(f"Exported {len(stats.runs)} runs to {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="STS2 Act 1 Simulator — pure algorithmic strategy testing"
    )
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of runs to simulate (default: 100)")
    parser.add_argument("--character", type=str, default="IRONCLAD",
                        help="Character ID (default: IRONCLAD)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base random seed (default: random)")
    parser.add_argument("--solver-time", type=float, default=200.0,
                        help="Solver time limit per turn in ms (default: 200)")
    parser.add_argument("--csv", type=str, default=None,
                        help="Export results to CSV file")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-floor progress")
    parser.add_argument("--no-progress", action="store_true",
                        help="Suppress progress bar")
    args = parser.parse_args()

    print(f"STS2 Act 1 Simulator")
    print(f"  Character: {args.character}")
    print(f"  Runs: {args.runs}")
    print(f"  Solver time limit: {args.solver_time}ms/turn")
    if args.seed is not None:
        print(f"  Base seed: {args.seed}")
    print()

    stats = run_batch(
        num_runs=args.runs,
        character=args.character,
        base_seed=args.seed,
        solver_time_limit_ms=args.solver_time,
        verbose=args.verbose,
        progress=not args.no_progress,
    )

    print_stats(stats)

    if args.csv:
        export_csv(stats, args.csv)


if __name__ == "__main__":
    main()
