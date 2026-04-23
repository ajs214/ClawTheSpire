"""Full Act 1 run training for AlphaZero.

Plays complete runs using MCTS for combat + deterministic advisor for
non-combat decisions. The network learns HP conservation across combats
and plays with naturally evolving decks.

Value targets: based on floor reached + HP remaining, giving continuous
signal across the full run. Early combats where the player took too
much damage get lower values because the run died later.
"""

from __future__ import annotations

import copy
import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch

from ..actions import Action, END_TURN, enumerate_actions
from ..combat_engine import (
    can_play_card,
    end_combat_relics,
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_combat,
    start_turn,
    tick_enemy_powers,
)
from ..card_picker import classify_deck
from ..constants import CardType
from ..data_loader import CardDB, load_cards
from ..models import Card, CombatState, EnemyState, PlayerState
from ..simulator import (
    _ensure_data_loaded,
    _CHARACTERS_BY_ID,
    _ACTS_BY_ID,
    _ENCOUNTERS_BY_ID,
    _spawn_enemy,
    _create_enemy_ai,
    _set_enemy_intents,
    _resolve_sim_intents,
    _generate_act1_map,
    _generate_act1_map_with_choices,
    _pick_encounter,
    _build_card_pool,
    _offer_card_rewards,
    _pick_card_reward,
    _rest_site_decision,
    _simulate_event,
    _simulate_shop,
    _normalize_card_id,
    GOLD_REWARDS,
    POTION_DROP_CHANCE,
    POTION_SLOTS,
    POTION_TYPES,
    _random_potion,
    SHOP_CARD_REMOVE_COST,
    SHOP_CARD_COSTS,
    SHOP_POTION_COST,
    SHOP_RELIC_COST,
    _get_shop_relic_pool,
)

from .encoding import EncoderConfig, Vocabs, compute_relic_synergy_features
from .mcts import MCTS, scale_simulations
from .network import STS2Network
from .self_play import (
    TrainingSample, OptionSample,
    OPTION_REST, OPTION_SMITH, OPTION_SHOP_REMOVE, OPTION_SHOP_BUY,
    OPTION_SHOP_LEAVE, OPTION_CARD_REWARD, OPTION_CARD_SKIP,
    OPTION_SHOP_BUY_POTION, OPTION_EVENT_CHOICE, OPTION_SHOP_BUY_RELIC,
    ROOM_TYPE_TO_OPTION,
    _affordable_play_actions,
)
from . import shadow_advisor as _shadow
from ..effects import discard_card_from_hand
from .. import relic_effects
from .state_tensor import encode_state, encode_actions

# Relic IDs that are class-locked to non-Silent characters and must never
# be offered to the Silent training pool.  Burning Blood / Black Blood
# are Ironclad starter/upgrade, the others are defect/regent/necrobinder
# starters or upgrades.
_NON_SILENT_RELIC_IDS: set[str] = {
    "BURNING_BLOOD", "BLACK_BLOOD",            # Ironclad
    "CRACKED_CORE", "FROZEN_CORE",             # Defect
    "PURE_WATER", "GLASS_MARBLE",              # Watcher / Defect variants
    "NECRONOMICURSE",                          # Necrobinder flavour
}

# Build the full Silent-relevant relic pool from the relic_effects
# registry.  Any relic with a behavioural entry anywhere in
# relic_effects.py is eligible to drop except the class-locked ones
# above.  The full run and mcts_combat never award the starter Ring of
# the Snake (granted in setup) so we strip it from the drop pool too.
def _build_silent_relic_pool() -> list[str]:
    pool = sorted(
        rid for rid in relic_effects.simulated_relic_ids()
        if rid not in _NON_SILENT_RELIC_IDS
        and rid != "RING_OF_THE_SNAKE"
    )
    return pool


IMPLEMENTED_RELIC_POOL = _build_silent_relic_pool()
# Alias for backwards compatibility with any external callers.
ELITE_RELIC_POOL = IMPLEMENTED_RELIC_POOL

# STS relic ID → display name, used to bridge combat_engine's ID-keyed
# relics set to relic_synergy.score_relic_for_deck's name-keyed scorer.
_RELIC_ID_TO_NAME: dict[str, str] = {
    "ANCHOR":             "Anchor",
    "ART_OF_WAR":         "Art of War",
    "BAG_OF_MARBLES":     "Bag of Marbles",
    "BAG_OF_PREPARATION": "Bag of Preparation",
    "BLOOD_VIAL":         "Blood Vial",
    "BRONZE_SCALES":      "Bronze Scales",
    "CLOAK_CLASP":        "Cloak Clasp",
    "FESTIVE_POPPER":     "Festive Popper",
    "KUNAI":              "Kunai",
    "LANTERN":            "Lantern",
    "LETTER_OPENER":      "Letter Opener",
    "MEAT_ON_THE_BONE":   "Meat on the Bone",
    "NUNCHAKU":           "Nunchaku",
    "ODDLY_SMOOTH_STONE": "Oddly Smooth Stone",
    "ORNAMENTAL_FAN":     "Ornamental Fan",
    "SHURIKEN":           "Shuriken",
    "STRIKE_DUMMY":       "Strike Dummy",
}


def _pick_best_relic(
    pool: list[str],
    deck: list[Card],
    owned: set[str],
    rng: random.Random,
    sample_size: int = 3,
) -> tuple[str | None, list[str]]:
    """Draw up to ``sample_size`` relics from ``pool`` and return the
    one that scores highest against the current deck, plus the full set
    of offered options.

    Mirrors the in-game 1-of-3 drop choice the player would see, and
    uses :func:`relic_synergy.score_relic_for_deck` so the pick is
    deck-aware (prefers Kunai for shiv decks, Art of War for heavy
    skill decks, etc).  Returns ``(None, [])`` if the pool is empty or
    the player already owns everything.  Otherwise returns
    ``(best_relic_id, offered_list)`` where ``offered_list`` includes
    the picked relic and all alternatives offered.
    """
    eligible = [r for r in pool if r not in owned]
    if not eligible:
        return None, []
    sample = rng.sample(eligible, min(sample_size, len(eligible)))
    try:
        from ..relic_synergy import score_relic_for_deck
        best_id, best_score = sample[0], float("-inf")
        for rid in sample:
            display = _RELIC_ID_TO_NAME.get(rid, rid)
            score = score_relic_for_deck(display, deck)
            if score > best_score:
                best_id, best_score = rid, score
        return best_id, sample
    except Exception:
        # If scoring blows up for any reason, fall back to a random
        # pick so we never silently drop a relic grant.
        chosen = rng.choice(sample)
        return chosen, sample


# Character starter relics
STARTER_RELICS = {
    "SILENT": "RING_OF_THE_SNAKE",
    "IRONCLAD": "BURNING_BLOOD",
}


def _apply_relic_pickup(
    relic_id: str,
    *,
    hp: int,
    max_hp: int,
    gold: int,
    potions: list[dict],
    potion_slots: int,
) -> tuple[int, int, int, int]:
    """Apply out-of-combat relic pickup effects (max HP / gold / potions / slots).

    Returns updated ``(hp, max_hp, gold, potion_slots)``.  ``potions`` is
    mutated in place when the relic adds free potions or slots.  Starter
    relics (Ring of the Snake / Burning Blood) also flow through here
    and are harmless because they aren't in any pickup table.
    """
    # Max HP bonuses are applied additively and also top up current HP
    # so the player can't lose effective HP on pickup.
    mhp_bonus = relic_effects.MAX_HP_ON_PICKUP.get(relic_id)
    if mhp_bonus:
        max_hp += mhp_bonus
        hp = min(hp + mhp_bonus, max_hp)

    # One-off gold bonuses.
    gold_bonus = relic_effects.GOLD_ON_PICKUP.get(relic_id)
    if gold_bonus:
        gold += gold_bonus

    # Potion slots.
    slot_bonus = relic_effects.POTION_SLOTS_ON_PICKUP.get(relic_id)
    if slot_bonus:
        potion_slots += slot_bonus

    return hp, max_hp, gold, potion_slots


def _grant_relic(
    relics: set[str],
    relic_id: str,
    *,
    hp: int,
    max_hp: int,
    gold: int,
    potions: list[dict],
    potion_slots: int,
) -> tuple[int, int, int, int]:
    """Add a relic to the owned set and apply its pickup effects.

    Returns the updated ``(hp, max_hp, gold, potion_slots)`` tuple so
    callers can rebind their own locals.
    """
    if relic_id in relics:
        return hp, max_hp, gold, potion_slots
    relics.add(relic_id)
    return _apply_relic_pickup(
        relic_id,
        hp=hp, max_hp=max_hp, gold=gold,
        potions=potions, potion_slots=potion_slots,
    )


# ---------------------------------------------------------------------------
# OOC Relic Effect Helpers
# ---------------------------------------------------------------------------

def _upgrade_card(card: Card, card_db: CardDB) -> None:
    """Upgrade a card in place using card_db, or fall back to approximation."""
    if card.upgraded:
        return
    try:
        upgraded = card_db.get_upgraded(card.id)
        if upgraded:
            card.damage = upgraded.damage
            card.block = upgraded.block
            card.cards_draw = upgraded.cards_draw
            card.energy_gain = upgraded.energy_gain
            card.hp_loss = upgraded.hp_loss
            card.hit_count = upgraded.hit_count
            card.powers_applied = upgraded.powers_applied
            card.cost = upgraded.cost
            card.upgraded = True
            return
    except Exception:
        pass
    # Fallback: approximate upgrade by bumping damage/block by 25%
    card.upgraded = True
    if card.damage and card.damage > 0:
        card.damage += max(1, card.damage // 4)
    if card.block and card.block > 0:
        card.block += max(1, card.block // 4)


def _remove_worst_cards(deck: list[Card], count: int) -> None:
    """Remove the worst cards from deck (Strikes first, then Defends)."""
    for _ in range(min(count, len(deck))):
        # Prioritize removing Strikes, then Defends, then first card
        strikes = [i for i, c in enumerate(deck) if "Strike" in c.name]
        if strikes:
            deck.pop(strikes[0])
            continue
        defends = [i for i, c in enumerate(deck) if "Defend" in c.name]
        if defends:
            deck.pop(defends[0])
            continue
        if deck:
            deck.pop(0)


def _transform_starters(deck: list[Card], card_db: CardDB) -> None:
    """Replace all Strikes and Defends with random non-starter cards."""
    try:
        all_cards = list(card_db.all_cards())
        non_starters = [
            c for c in all_cards
            if "Strike" not in c.name and "Defend" not in c.name
            and c.card_type in (CardType.ATTACK, CardType.SKILL, CardType.POWER)
            and c.cost >= 0
        ]
    except Exception:
        return

    indices_to_replace = [
        i for i, c in enumerate(deck)
        if "Strike" in c.name or "Defend" in c.name
    ]
    for idx in indices_to_replace:
        if non_starters:
            replacement = copy.copy(random.choice(non_starters))
            deck[idx] = replacement


def _apply_relic_ooc_effects(
    relic_id: str,
    deck: list[Card],
    card_db: CardDB,
    hp: int,
    max_hp: int,
) -> tuple[int, int]:
    """Apply out-of-combat relic effects that modify the deck.

    Returns updated (hp, max_hp) tuple.
    """
    if relic_id == "WHETSTONE":
        # Upgrade 2 random Attacks in deck
        attacks = [c for c in deck if c.card_type == CardType.ATTACK and not c.upgraded]
        for c in random.sample(attacks, min(2, len(attacks))):
            _upgrade_card(c, card_db)

    elif relic_id == "WAR_PAINT":
        # Upgrade 2 random Skills
        skills = [c for c in deck if c.card_type == CardType.SKILL and not c.upgraded]
        for c in random.sample(skills, min(2, len(skills))):
            _upgrade_card(c, card_db)

    elif relic_id == "YUMMY_COOKIE":
        # Upgrade 4 random cards
        upgradeable = [c for c in deck if not c.upgraded]
        for c in random.sample(upgradeable, min(4, len(upgradeable))):
            _upgrade_card(c, card_db)

    elif relic_id == "SAND_CASTLE":
        # Upgrade 6 random cards
        upgradeable = [c for c in deck if not c.upgraded]
        for c in random.sample(upgradeable, min(6, len(upgradeable))):
            _upgrade_card(c, card_db)

    elif relic_id == "POMANDER":
        # Upgrade 1 random card
        upgradeable = [c for c in deck if not c.upgraded]
        if upgradeable:
            _upgrade_card(random.choice(upgradeable), card_db)

    elif relic_id == "FRAGRANT_MUSHROOM":
        # Lose 15 HP, upgrade 3 random cards
        hp = max(1, hp - 15)
        upgradeable = [c for c in deck if not c.upgraded]
        for c in random.sample(upgradeable, min(3, len(upgradeable))):
            _upgrade_card(c, card_db)

    elif relic_id == "PANDORAS_BOX":
        # Transform ALL Strikes and Defends into random cards
        _transform_starters(deck, card_db)

    elif relic_id == "EMPTY_CAGE":
        # Remove 2 cards (remove worst — Strikes/Defends first)
        _remove_worst_cards(deck, 2)

    elif relic_id == "PRECISE_SCISSORS":
        _remove_worst_cards(deck, 1)

    elif relic_id == "PRECARIOUS_SHEARS":
        hp = max(1, hp - 13)
        _remove_worst_cards(deck, 2)

    elif relic_id == "PRESERVED_FOG":
        _remove_worst_cards(deck, 5)
        # Should add Folly curse but we don't have curse cards — skip

    elif relic_id == "GHOST_SEED":
        # Strikes and Defends gain Ethereal
        for c in deck:
            if "Strike" in c.name or "Defend" in c.name:
                if "Ethereal" not in c.keywords:
                    c.keywords = c.keywords | frozenset({"Ethereal"})

    return hp, max_hp


# ---------------------------------------------------------------------------
# MCTS-based combat within a full run
# ---------------------------------------------------------------------------

def mcts_combat(
    deck: list[Card],
    player_hp: int,
    player_max_hp: int,
    player_max_energy: int,
    encounter_id: str,
    card_db: CardDB,
    mcts: MCTS,
    vocabs: Vocabs,
    config: EncoderConfig,
    rng: random.Random,
    mcts_simulations: int = 100,
    temperature: float = 1.0,
    max_turns: int = 30,
    potions: list[dict] | None = None,
    relics: frozenset[str] | None = None,
    is_boss: bool = False,
    is_elite: bool = False,
    detail_log: bool = False,
    tea_set_bonus: int = 0,
) -> tuple[list[TrainingSample], str, int, int, list[dict], dict | None]:
    """Run one combat using MCTS.

    Returns (samples, outcome, turns, hp_after, remaining_potions, detail).
    `detail` is a rich per-turn log populated when `detail_log=True`
    (currently used for boss fights so we can analyse play patterns).
    It is None otherwise to keep overhead zero for normal combats.
    """
    _ensure_data_loaded()

    enc = _ENCOUNTERS_BY_ID.get(encounter_id, {})
    monster_list = enc.get("monsters", [])
    enemies: list[EnemyState] = []
    enemy_ais = []
    for m in monster_list:
        mid = m["id"]
        enemies.append(_spawn_enemy(mid))
        enemy_ais.append(_create_enemy_ai(mid))

    if not enemies:
        return [], "win", 0, player_hp, potions or [], None

    draw_pile = list(deck)
    rng.shuffle(draw_pile)

    # Apply tea set bonus before combat starts
    combat_max_energy = player_max_energy
    if tea_set_bonus > 0:
        combat_max_energy += tea_set_bonus

    player = PlayerState(
        hp=player_hp, max_hp=player_max_hp,
        energy=combat_max_energy, max_energy=combat_max_energy,
        draw_pile=draw_pile,
        potions=[dict(p) for p in (potions or [])],
    )
    state = CombatState(player=player, enemies=enemies, relics=relics or frozenset())
    start_combat(state, is_elite=is_elite, is_boss=is_boss)

    # Apply pre-combat relic effects (after start_combat so relics are in state)
    if relics:
        relics_set = set(relics) if not isinstance(relics, set) else relics

        # BELLOWS: upgrade starting hand
        if "BELLOWS" in relics_set:
            for c in state.player.hand:
                _upgrade_card(c, card_db)

        # STONE_CRACKER: at boss start, upgrade 3 random draw pile cards
        if is_boss and "STONE_CRACKER" in relics_set:
            upgradeable = [c for c in state.player.draw_pile if not c.upgraded]
            for c in random.sample(upgradeable, min(3, len(upgradeable))):
                _upgrade_card(c, card_db)

        # RINGING_TRIANGLE: retain hand on turn 1
        # (implementation deferred to combat turn 1 logic if needed)

        # VEXING_PUZZLEBOX: random card from draw pile to hand, cost 0
        if "VEXING_PUZZLEBOX" in relics_set and state.player.draw_pile:
            chosen = random.choice(state.player.draw_pile)
            state.player.draw_pile.remove(chosen)
            free_copy = copy.copy(chosen)
            free_copy.cost = 0
            state.player.hand.append(free_copy)

        # JEWELED_MASK: find Power in draw pile, move to hand, free
        if "JEWELED_MASK" in relics_set:
            powers = [c for c in state.player.draw_pile if c.card_type == CardType.POWER]
            if powers:
                chosen = random.choice(powers)
                state.player.draw_pile.remove(chosen)
                free_copy = copy.copy(chosen)
                free_copy.cost = 0
                state.player.hand.append(free_copy)

        # CHOICES_PARADOX: random card to hand with Retain
        if "CHOICES_PARADOX" in relics_set and state.player.draw_pile:
            chosen = random.choice(state.player.draw_pile)
            state.player.draw_pile.remove(chosen)
            free_copy = copy.copy(chosen)
            free_copy.keywords = chosen.keywords | frozenset({"Retain"})
            state.player.hand.append(free_copy)

        # GAMBLING_CHIP: discard hand and redraw
        if "GAMBLING_CHIP" in relics_set:
            hand_size = len(state.player.hand)
            state.player.discard_pile.extend(state.player.hand)
            state.player.hand.clear()
            from ..effects import draw_cards
            draw_cards(state, hand_size)

        # FAKE_SNECKO_EYE: randomize card costs (Confused)
        if "FAKE_SNECKO_EYE" in relics_set:
            for c in state.player.hand + state.player.draw_pile:
                if c.cost >= 0:
                    c.cost = random.randint(0, 3)

    # --- Initialise detail log (boss fights use this) ---
    detail: dict | None = None
    if detail_log:
        from collections import Counter as _Counter
        _deck_counts = _Counter(c.id for c in deck)
        detail = {
            "encounter_id": encounter_id,
            "is_boss": is_boss,
            "monster_ids": [m.get("id") for m in monster_list],
            "monster_start_hp": [e.hp for e in enemies],
            "monster_max_hp": [e.max_hp for e in enemies],
            "player_start_hp": player_hp,
            "player_max_hp": player_max_hp,
            "player_max_energy": player_max_energy,
            "deck_size": len(deck),
            "deck_counts": dict(_deck_counts),
            "relics": sorted(list(relics)) if relics else [],
            "potions_at_start": [p.get("id") if isinstance(p, dict) else None
                                 for p in (potions or [])],
            "turns": [],
            "outcome": None,
            "final_player_hp": None,
            "total_turns": 0,
            "potions_used": [],
        }

    # Boss fights: dump all offensive potions pre-combat and heal if low.
    # This front-loads burst damage/buffs when it matters most.
    # Non-boss fights: MCTS decides potion usage organically (save for boss).
    if is_boss and state.player.potions:
        from ..simulator import _use_precombat_potions, _use_emergency_potion
        pre_potions = [p for p in state.player.potions if p]
        _potions_before_dump = [p.get("id") for p in pre_potions if p]
        remaining = _use_precombat_potions(state, pre_potions)
        if state.player.hp < state.player.max_hp * 0.40:
            remaining = _use_emergency_potion(state, remaining)
        # Update the player's potion list (keep slots, clear used ones)
        state.player.potions = remaining
        if detail is not None:
            _after_ids = [p.get("id") for p in remaining if p]
            _used = [pid for pid in _potions_before_dump if pid not in _after_ids]
            detail["precombat_potions_used"] = _used
            detail["potions_used"].extend(_used)

    samples: list[TrainingSample] = []
    turn_sample_ranges: list[tuple[int, int]] = []  # (start_idx, end_idx) per turn
    outcome = None
    _lizard_tail_used = False  # Track one-time LIZARD_TAIL use

    for turn_num in range(1, max_turns + 1):
        start_turn(state)
        _set_enemy_intents(state, enemy_ais)

        # --- Per-turn detail snapshot ---
        turn_record: dict | None = None
        if detail is not None:
            turn_record = {
                "turn": turn_num,
                "energy_start": state.player.energy,
                "hand_start": [c.id for c in state.player.hand],
                "draw_pile_size": len(state.player.draw_pile),
                "discard_pile_size": len(state.player.discard_pile),
                "hp_start": state.player.hp,
                "enemy_hp_start": [e.hp for e in state.enemies],
                "enemy_intents": [
                    {"type": e.intent_type, "dmg": e.intent_damage, "hits": e.intent_hits}
                    for e in state.enemies
                ],
                "cards_played": [],
                "targets": [],
                "potions_used": [],
                "wasted_end_turn": False,
                "forced_overrides": 0,
            }

        turn_start_sample = len(samples)
        cards_this_turn = 0
        while cards_this_turn < 12:
            outcome = is_combat_over(state)
            if outcome:
                break

            actions = enumerate_actions(state)
            if not actions:
                break

            state_tensors = encode_state(state, vocabs, config)
            action_card_ids, action_features, action_mask = encode_actions(actions, state, vocabs, config)

            scaled_sims = scale_simulations(mcts_simulations, len(actions), is_boss=is_boss)
            action, policy, _root_value, _mcts_actions = mcts.search(
                state, num_simulations=scaled_sims,
                temperature=temperature,
            )

            # Force-play override: if MCTS wants END_TURN but affordable
            # non-junk cards exist, override 80% of the time to generate
            # exploration data showing the value of playing those cards.
            wasted = False
            if action.action_type == "end_turn":
                affordable = _affordable_play_actions(actions, state)
                if affordable:
                    wasted = True
                    if turn_record is not None:
                        turn_record["wasted_end_turn"] = True
                    if rng.random() < 0.80:
                        action = rng.choice(affordable)
                        if turn_record is not None:
                            turn_record["forced_overrides"] += 1

            samples.append(TrainingSample(
                state_tensors=state_tensors,
                policy=policy,
                value=0.0,  # Filled after run ends
                action_card_ids=action_card_ids,
                action_features=action_features,
                action_mask=action_mask,
                num_actions=len(actions),
                wasted_energy=wasted,
            ))

            if action.action_type == "end_turn":
                break

            if action.action_type == "choose_card":
                # Resolve pending choice (discard, etc.) — doesn't count as a card play
                if action.choice_idx is not None and state.pending_choice is not None:
                    pc = state.pending_choice
                    if pc.choice_type == "discard_from_hand":
                        if action.choice_idx < len(state.player.hand):
                            discard_card_from_hand(state, action.choice_idx)
                        pc.chosen_so_far.append(action.choice_idx)
                        if len(pc.chosen_so_far) >= pc.num_choices:
                            state.pending_choice = None
            elif action.action_type == "use_potion":
                from ..combat_engine import use_potion as _use_potion
                if action.potion_idx is not None:
                    if turn_record is not None:
                        # Capture the potion id before it's consumed
                        try:
                            _pot = state.player.potions[action.potion_idx]
                            if _pot:
                                turn_record["potions_used"].append(_pot.get("id"))
                                detail["potions_used"].append(_pot.get("id"))
                        except Exception:
                            pass
                    _use_potion(state, action.potion_idx)
                cards_this_turn += 1  # count toward safety cap
            elif action.card_idx is not None and can_play_card(state, action.card_idx):
                if turn_record is not None:
                    try:
                        _card = state.player.hand[action.card_idx]
                        turn_record["cards_played"].append(_card.id)
                        turn_record["targets"].append(action.target_idx)
                    except Exception:
                        pass
                play_card(state, action.card_idx, action.target_idx, card_db)
                cards_this_turn += 1

            outcome = is_combat_over(state)
            if outcome:
                break

        turn_end_sample = len(samples)
        turn_sample_ranges.append((turn_start_sample, turn_end_sample))

        # Snapshot post-player-phase (before enemy phase resolves)
        if turn_record is not None:
            turn_record["enemy_hp_after_player"] = [e.hp for e in state.enemies]
            turn_record["damage_dealt"] = sum(
                max(0, before - after)
                for before, after in zip(
                    turn_record["enemy_hp_start"],
                    turn_record["enemy_hp_after_player"],
                )
            )
            turn_record["block_played"] = state.player.block
            # If combat ended during player phase, log and append.
            if outcome:
                turn_record["hp_end"] = state.player.hp
                turn_record["damage_taken"] = 0
                turn_record["enemy_hp_end"] = [e.hp for e in state.enemies]
                turn_record["ended_combat"] = True
                detail["turns"].append(turn_record)
                turn_record = None

        if outcome:
            break

        # -- Intent-aware reward signals --
        # Compute expected incoming damage from enemy intents
        incoming_damage = 0
        for e in state.enemies:
            if e.is_alive and e.intent_type == "Attack" and e.intent_damage:
                incoming_damage += e.intent_damage * max(1, e.intent_hits)

        # Snapshot state before enemy phase
        hp_before_enemy = state.player.hp
        player_block_played = state.player.block  # block accumulated this turn
        hand_block_available = sum(
            c.block for c in state.player.hand
            if c.block and c.block > 0 and c.cost <= state.player.energy
        )
        hand_damage_available = sum(
            c.damage for c in state.player.hand
            if c.damage and c.damage > 0 and c.cost <= state.player.energy
        )

        # Signal 1: Offensive-when-safe — penalise block plays when
        # no enemies intend to attack (wasted tempo).
        if (incoming_damage == 0 and player_block_played > 0
                and turn_start_sample < turn_end_sample):
            # Scale by how much block was wasted relative to energy
            safe_block_penalty = min(0.08, player_block_played / max(1, player_max_hp) * 0.3)
            for idx in range(turn_start_sample, turn_end_sample):
                samples[idx].value_penalty += safe_block_penalty

        # Signal 2: Lethal-awareness — penalise not killing an enemy
        # that could have been killed this turn (they'll deal more
        # damage on future turns if left alive).
        for e in state.enemies:
            if e.is_alive and e.hp > 0 and e.hp <= hand_damage_available:
                # An enemy was killable but survived
                lethal_penalty = min(0.10, e.hp / max(1, player_max_hp) * 0.25)
                for idx in range(turn_start_sample, turn_end_sample):
                    samples[idx].value_penalty += lethal_penalty
                break  # only penalise for one missed lethal per turn

        end_turn(state)
        resolve_enemy_intents(state)
        _resolve_sim_intents(state, enemy_ais)
        tick_enemy_powers(state)

        # LIZARD_TAIL: prevent death (one-time use)
        if state.player.hp <= 0 and "LIZARD_TAIL" in (relics or frozenset()) and not _lizard_tail_used:
            state.player.hp = max(1, state.player.max_hp // 2)
            _lizard_tail_used = True

        hp_after_enemy = state.player.hp
        damage_taken = max(0, hp_before_enemy - hp_after_enemy)

        # Finalise this turn's detail record
        if turn_record is not None:
            turn_record["hp_end"] = state.player.hp
            turn_record["damage_taken"] = damage_taken
            turn_record["incoming_damage"] = incoming_damage
            turn_record["enemy_hp_end"] = [e.hp for e in state.enemies]
            detail["turns"].append(turn_record)
            turn_record = None

        # Signal 3: Intent-weighted blocking — penalise based on how
        # far actual block was from optimal (matching incoming damage).
        # Under-blocking penalises harder than over-blocking.
        #
        # Both directions now skip the penalty when the player had no
        # real choice: under-block is only "wrong" if enough affordable
        # block was in hand to actually close the gap, and over-block
        # is only "wrong" if there were attacks the player could have
        # played instead. We don't punish forced decisions.
        if turn_start_sample < turn_end_sample and incoming_damage > 0:
            if player_block_played < incoming_damage:
                # Under-blocked: took avoidable damage
                gap = incoming_damage - player_block_played
                under_penalty = min(0.15, gap / max(1, player_max_hp) * 0.4)
                # Only penalise if they had enough affordable block in
                # hand to fully close the gap. If total possible block
                # was less than incoming damage, the under-block was
                # forced by deck/energy, not a bad decision.
                if hand_block_available >= gap:
                    for idx in range(turn_start_sample, turn_end_sample):
                        samples[idx].value_penalty += under_penalty
            elif player_block_played > incoming_damage * 1.5:
                # Over-blocked by >50%: wasted resources on defence —
                # but only if there was something else to do. If the
                # player had no affordable attacks left in hand, then
                # over-blocking was the only legal play and shouldn't
                # be punished.
                if hand_damage_available > 0:
                    excess = player_block_played - incoming_damage
                    over_penalty = min(0.06, excess / max(1, player_max_hp) * 0.15)
                    for idx in range(turn_start_sample, turn_end_sample):
                        samples[idx].value_penalty += over_penalty

        outcome = is_combat_over(state)
        if outcome:
            break

    if outcome is None:
        outcome = "lose"

    hp_after = max(0, state.player.hp) if outcome == "win" else 0
    remaining_potions = [p for p in state.player.potions if p]

    # Finalise detail log
    if detail is not None:
        detail["outcome"] = outcome
        detail["final_player_hp"] = hp_after
        detail["total_turns"] = turn_num
        detail["final_enemy_hp"] = [e.hp for e in state.enemies]

    return samples, outcome, turn_num, hp_after, remaining_potions, detail


# ---------------------------------------------------------------------------
# Network-based card reward selection
# ---------------------------------------------------------------------------

def _network_pick_card(
    offered: list[Card],
    deck: list[Card],
    hp: int,
    max_hp: int,
    floor: int,
    mcts: MCTS,
    vocabs: Vocabs,
    config: EncoderConfig,
    card_db: CardDB,
    relics: frozenset[str] | set[str] | None = None,
    *,
    organic_warm_start: bool = False,
) -> tuple[Card | None, OptionSample | None]:
    """Pick a card reward via the option-head network (IMPROVEMENTS.md #3).

    Historical behavior: the organic ``card_picker.pick_card`` made the
    actual decision and the network's option head was trained via
    supervised learning on the organic picker's output. That was
    imitation learning, not self-play — the network's ceiling was
    exactly the heuristic's ceiling.

    New behavior (self-play): the network's option head makes the pick
    directly via ``pick_best_option``. The resulting ``OptionSample``
    gets its ``value`` field filled in by ``_assign_run_values`` at run
    end, so the option head learns from actual run outcomes instead of
    mimicking the heuristic.

    ``organic_warm_start`` switches back to the old imitation mode.
    Set True during the first generation of a fresh training run to
    hand the network a reasonable starting policy; flip to False after
    the network's agreement rate with the organic picker climbs past
    ~70% (measured via ``tools/agreement_rate.py``).

    Returns (picked_card_or_None, training_sample_or_None).
    """
    if not offered:
        return None, None

    sample = None
    pick: Card | None = None
    try:
        network = mcts.network
        player = PlayerState(
            hp=hp, max_hp=max_hp, energy=3, max_energy=3,
            hand=[], draw_pile=list(deck),
        )
        dummy_state = CombatState(player=player, enemies=[], turn=0, floor=floor)

        state_tensors = encode_state(dummy_state, vocabs, config)
        state_tensors = {k: v.to(mcts.device) for k, v in state_tensors.items()}

        opt_types = [OPTION_CARD_REWARD] * len(offered) + [OPTION_CARD_SKIP]
        opt_cards = []
        for card in offered:
            base_id = card.id.rstrip("+")
            opt_cards.append(vocabs.cards.get(base_id))
        opt_cards.append(0)  # PAD for skip

        # Build deck card vocab IDs for the dedicated card_eval_head.
        # Exclude unupgraded starter cards so the deck_summary reflects
        # only drafted cards — otherwise 12 base cards drown out the
        # signal from the 1-3 cards actually picked.
        _BASE_CARD_IDS = {"STRIKE_SILENT", "DEFEND_SILENT", "NEUTRALIZE", "SURVIVOR"}
        deck_card_ids = []
        for c in deck:
            base_id = c.id.rstrip("+")
            is_upgraded = c.id.endswith("+")
            if base_id in _BASE_CARD_IDS and not is_upgraded:
                continue  # skip unmodified starter card
            deck_card_ids.append(vocabs.cards.get(base_id))  # UNK_IDX if missing

        # Shadow pick = what the organic heuristic would have chosen.
        # Now that the network makes the actual pick, this is a real
        # agreement signal (not self-comparison) and surfaces in the
        # agreement-rate diagnostic (IMPROVEMENTS.md #10).
        shadow_idx = _shadow.shadow_pick_card_reward(
            offered, deck, hp, max_hp, floor, relics=relics,
        )

        _scores = None  # populated in self-play mode for score-spread diagnostics
        if organic_warm_start:
            # Imitation mode: defer to the organic picker for the
            # actual choice. Useful for seeding a fresh network with
            # a sensible starting policy. Turn this off once the
            # network has learned enough to drive itself.
            from ..card_picker import pick_card as organic_pick
            pick = organic_pick(offered, deck, floor, hp, max_hp, relics=relics)
            if pick is None:
                chosen_idx = len(offered)  # skip slot
            else:
                chosen_idx = next(
                    (i for i, c in enumerate(offered) if c.id == pick.id),
                    len(offered),
                )
        else:
            # Self-play mode: the dedicated card_eval_head picks,
            # using deck composition and relic synergy for context.
            relic_id_list = None
            relic_mask_list = None
            syn_feats = None
            if relics:
                relic_id_list = [vocabs.relics.get(r) for r in relics]
                relic_mask_list = [False] * len(relic_id_list)  # all valid
                syn_feats = compute_relic_synergy_features(relics)
            with torch.no_grad():
                hidden = network.encode_state(**state_tensors)
                best_idx, _scores = network.pick_best_card(
                    hidden, deck_card_ids, opt_types, opt_cards,
                    relic_ids=relic_id_list, relic_mask=relic_mask_list,
                    synergy_features=syn_feats)
            chosen_idx = int(best_idx)
            if chosen_idx >= len(offered):
                pick = None  # skip slot
            else:
                pick = offered[chosen_idx]

        sample = OptionSample(
            state_tensors={k: v.cpu() for k, v in state_tensors.items()},
            option_types=opt_types,
            option_cards=opt_cards,
            chosen_idx=chosen_idx,
            value=0.0,  # Filled after run ends by _assign_run_values
            shadow_chosen_idx=shadow_idx,
            deck_card_ids=deck_card_ids,
            pick_scores=_scores if not organic_warm_start else None,
            relic_ids=relic_id_list if not organic_warm_start else None,
            relic_mask=relic_mask_list if not organic_warm_start else None,
            synergy_features=syn_feats if not organic_warm_start else None,
        )
    except Exception as _card_err:
        # On any failure, fall back to the organic picker with no
        # training sample. This keeps training runs from crashing on
        # edge cases the network hasn't seen.
        print(f"  [card-pick] fallback: {_card_err}", flush=True)
        try:
            from ..card_picker import pick_card as organic_pick
            pick = organic_pick(offered, deck, floor, hp, max_hp, relics=relics)
        except Exception:
            pick = offered[0] if offered else None
        sample = None

    return pick, sample


# ---------------------------------------------------------------------------
# Full Act 1 run with MCTS combat
# ---------------------------------------------------------------------------

@dataclass
class FullRunResult:
    outcome: str  # "win" or "lose"
    floor_reached: int
    final_hp: int
    max_hp: int
    combats_won: int
    combats_fought: int
    deck_size: int
    samples: list[TrainingSample]
    deck_samples: list  # OptionSample list (card rewards, routed through option head)
    option_samples: list  # OptionSample list (rest/map/shop)
    combat_log: list[dict]
    run_log: list[dict] | None = None  # Per-floor journey log for the viewer
    archetype: str = "undecided"     # emergent archetype of final deck
    archetype_commitment: float = 0.0  # 0.0–1.0
    boss_detail: dict | None = None  # Rich per-turn log of the boss combat
    final_deck: list[str] | None = None  # Card IDs in deck at boss-fight time
    final_relics: list[str] | None = None  # Relic IDs owned at end of run (V8)


def play_full_run(
    mcts: MCTS,
    card_db: CardDB,
    vocabs: Vocabs,
    config: EncoderConfig,
    character: str = "SILENT",
    mcts_simulations: int = 100,
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> FullRunResult:
    """Play a full Act 1 run. Returns result with training samples."""
    if rng is None:
        rng = random.Random()

    _ensure_data_loaded()

    # Character setup
    char_data = _CHARACTERS_BY_ID.get(character, {})
    hp = char_data.get("starting_hp", 70)
    max_hp = hp
    gold = char_data.get("starting_gold", 99)
    max_energy = char_data.get("max_energy", 3)

    # Build starting deck
    raw_deck_ids = char_data.get("starting_deck", [])
    deck: list[Card] = []
    for raw_id in raw_deck_ids:
        card_id = _normalize_card_id(raw_id)
        card = card_db.get(card_id) or card_db.get(raw_id)
        if card:
            deck.append(copy.copy(card))

    if not deck:
        # Fallback: Silent starter
        for name, count in [("STRIKE_SILENT", 5), ("DEFEND_SILENT", 5),
                            ("NEUTRALIZE", 1), ("SURVIVOR", 1)]:
            c = card_db.get(name)
            if c:
                deck.extend([copy.copy(c) for _ in range(count)])

    # Card exploration tracking: count how many times each card has been
    # picked across this run. Used for decaying exploration (Option D).
    _card_explore_counts: dict[str, int] = {}  # card_id → times picked in this run

    # Card pools
    char_color = char_data.get("color", "green")
    color_map = {"red": "ironclad", "green": "silent", "blue": "defect",
                 "purple": "necrobinder", "yellow": "regent"}
    card_color = color_map.get(char_color, char_color)
    pools = _build_card_pool(card_db, card_color)

    # Act data + map
    act_data = _ACTS_BY_ID.get("OVERGROWTH", {})
    room_sequence = _generate_act1_map_with_choices(rng)

    # Starter relic + dynamic potion slot cap (lifted by Potion Belt etc.)
    relics: set[str] = set()
    potion_slots_cap: int = POTION_SLOTS
    starter_relic = STARTER_RELICS.get(character)
    if starter_relic:
        relics.add(starter_relic)
        hp, max_hp, gold, potion_slots_cap = _apply_relic_pickup(
            starter_relic,
            hp=hp, max_hp=max_hp, gold=gold,
            potions=[], potion_slots=potion_slots_cap,
        )

    # Run state
    all_samples: list[TrainingSample] = []
    deck_change_samples: list = []
    option_samples: list = []
    combat_samples_by_floor: dict[int, list[TrainingSample]] = {}
    combat_hp_data: dict[int, tuple[int, int, int]] = {}  # floor -> (hp_before, hp_after, potions_used)
    boss_floors: set[int] = set()
    combat_log: list[dict] = []
    run_log: list[dict] = []  # Per-floor journey for the viewer
    combats_won = 0
    combats_fought = 0
    boss_detail_holder: dict | None = None
    potions: list[dict] = []
    seen_encounters: set[str] = set()
    events_list = list(act_data.get("events", []))
    rng.shuffle(events_list)
    event_idx = 0
    floor_reached = 0

    # IMPROVEMENTS.md #7: Neow blessing. Exposed to training as an event-choice
    # style OptionSample so the option head learns cold-start opening policy
    # from run outcomes instead of a hand-rolled priority list. Reuses
    # OPTION_EVENT_CHOICE so no new option-type slot is needed in the network
    # embedding table (no checkpoint break). The per-blessing vocab ids are
    # registered under the ``__neow__`` sentinel event id.
    try:
        from ..simulator import enumerate_neow_options

        neow_choices = enumerate_neow_options(
            deck, hp, max_hp, gold, card_db, rng)
        if neow_choices:
            chosen_neow_changes: dict | None = None
            try:
                network = mcts.network
                player = PlayerState(
                    hp=hp, max_hp=max_hp,
                    energy=3, max_energy=3,
                    draw_pile=list(deck),
                )
                dummy = CombatState(
                    player=player, enemies=[],
                    floor=0, gold=gold,
                    relics=frozenset(relics),
                )
                st = encode_state(dummy, vocabs, config)
                st = {k: v.to(mcts.device) for k, v in st.items()}

                opt_types = [OPTION_EVENT_CHOICE] * len(neow_choices)
                # V10: use real EVENT_CHOICE_VOCAB ids from the
                # dedicated event_choice_embed table.
                opt_cards = [c["vocab_id"] for c in neow_choices]

                with torch.no_grad():
                    hidden = network.encode_state(**st)
                    best_idx, _scores = network.pick_best_option(
                        hidden, opt_types, opt_cards)

                shadow_idx = _shadow.shadow_pick_neow(
                    hp, max_hp, gold, deck)

                option_samples.append(OptionSample(
                    state_tensors={k: v.cpu() for k, v in st.items()},
                    option_types=opt_types,
                    option_cards=opt_cards,
                    chosen_idx=int(best_idx), value=0.0,
                    shadow_chosen_idx=shadow_idx,
                ))
                chosen_neow_changes = neow_choices[int(best_idx)]["changes"]
            except Exception:
                shadow_idx = _shadow.shadow_pick_neow(
                    hp, max_hp, gold, deck)
                chosen_neow_changes = neow_choices[shadow_idx]["changes"]

            # Apply Neow effects atomically using the same event-changes
            # dict shape — keeps a single code path for "event-like" state
            # mutations.
            _c = chosen_neow_changes or {}
            max_hp += _c.get("max_hp_delta", 0)
            if max_hp < 1:
                max_hp = 1
            hp = max(1, min(hp + _c.get("hp_delta", 0), max_hp))
            gold = max(0, gold + _c.get("gold_delta", 0))
            for idx in sorted(_c.get("cards_removed", []), reverse=True):
                if 0 <= idx < len(deck):
                    deck.pop(idx)
            for card in _c.get("cards_added", []):
                # Apply egg auto-upgrades
                if "FROZEN_EGG" in relics and card.card_type == CardType.POWER:
                    _upgrade_card(card, card_db)
                elif "MOLTEN_EGG" in relics and card.card_type == CardType.ATTACK:
                    _upgrade_card(card, card_db)
                elif "TOXIC_EGG" in relics and card.card_type == CardType.SKILL:
                    _upgrade_card(card, card_db)

                # BING_BONG: add TWO copies instead of one
                if "BING_BONG" in relics:
                    deck.append(card)
                    deck.append(copy.copy(card))
                else:
                    deck.append(card)
            if _c.get("grants_relic"):
                granted, offered = _pick_best_relic(
                    IMPLEMENTED_RELIC_POOL, deck, relics, rng,
                )
                if granted is not None:
                    hp, max_hp, gold, potion_slots_cap = _grant_relic(
                        relics, granted,
                        hp=hp, max_hp=max_hp, gold=gold,
                        potions=potions, potion_slots=potion_slots_cap,
                    )
                    # Apply OOC relic effects
                    hp, max_hp = _apply_relic_ooc_effects(
                        granted, deck, card_db, hp, max_hp)
    except Exception:
        # If anything goes wrong during the Neow step, silently skip it
        # rather than failing the whole run — keeps training robust to
        # future refactors that touch enumerate_neow_options.
        pass

    # Track tea set bonus for next combat
    _tea_set_bonus: int = 0

    for floor_num, room_entry in enumerate(room_sequence, 1):
        floor_reached = floor_num

        # Resolve map choice nodes via network
        if isinstance(room_entry, list):
            try:
                network = mcts.network
                player = PlayerState(hp=hp, max_hp=max_hp, energy=3, max_energy=3,
                                     draw_pile=list(deck))
                dummy = CombatState(player=player, enemies=[], floor=floor_num, gold=gold, relics=frozenset(relics))
                st = encode_state(dummy, vocabs, config)
                st = {k: v.to(mcts.device) for k, v in st.items()}

                opt_types = [ROOM_TYPE_TO_OPTION[rt] for rt in room_entry]
                opt_cards = [0] * len(room_entry)

                with torch.no_grad():
                    hidden = network.encode_state(**st)
                    best_idx, scores = network.pick_best_option(
                        hidden, opt_types, opt_cards)

                shadow_idx = _shadow.shadow_pick_map(
                    room_entry, hp, max_hp, gold, len(deck), floor_num,
                )
                option_samples.append(OptionSample(
                    state_tensors={k: v.cpu() for k, v in st.items()},
                    option_types=opt_types, option_cards=opt_cards,
                    chosen_idx=best_idx, value=0.0,
                    shadow_chosen_idx=shadow_idx,
                ))
                room_type = room_entry[best_idx]
                run_log.append({
                    "floor": floor_num, "type": "map_choice",
                    "options": list(room_entry),
                    "chosen": room_type,
                })
            except Exception:
                room_type = rng.choice(room_entry)
                run_log.append({
                    "floor": floor_num, "type": "map_choice",
                    "options": list(room_entry),
                    "chosen": room_type,
                    "fallback": True,
                })
        else:
            room_type = room_entry

        if room_type in ("weak", "normal", "elite", "boss"):
            enc_id = _pick_encounter(act_data, room_type, rng, seen_encounters)
            if enc_id is None:
                continue

            _is_boss = (room_type == "boss")
            _is_elite = (room_type == "elite")
            potions_before = len([p for p in potions if p])
            samples, outcome, turns, hp_after, potions, combat_detail = mcts_combat(
                deck=deck, player_hp=hp, player_max_hp=max_hp,
                player_max_energy=max_energy, encounter_id=enc_id,
                card_db=card_db, mcts=mcts, vocabs=vocabs, config=config,
                rng=rng, mcts_simulations=mcts_simulations,
                temperature=temperature, potions=potions,
                relics=frozenset(relics),
                is_boss=_is_boss,
                is_elite=_is_elite,
                detail_log=_is_boss,  # only log boss combats (overhead)
                tea_set_bonus=_tea_set_bonus,
            )
            # Reset tea set bonus after combat
            _tea_set_bonus = 0
            if _is_boss and combat_detail is not None:
                boss_detail_holder = combat_detail  # captured into FullRunResult below
            potions_after = len([p for p in potions if p])
            potions_used = max(0, potions_before - potions_after)

            combats_fought += 1
            combat_samples_by_floor[floor_num] = samples
            combat_hp_data[floor_num] = (hp, hp_after, potions_used)
            if room_type == "boss":
                boss_floors.add(floor_num)
            all_samples.extend(samples)

            combat_log.append({
                "floor": floor_num, "encounter": enc_id,
                "room_type": room_type, "outcome": outcome,
                "turns": turns, "hp_before": hp, "hp_after": hp_after,
            })

            # Journey log: combat result
            run_log.append({
                "floor": floor_num, "type": "combat",
                "room_type": room_type, "encounter": enc_id,
                "outcome": outcome, "turns": turns,
                "hp_before": hp, "hp_after": hp_after,
                "potions_used": potions_used,
                "deck_snapshot": [c.id for c in deck],
                "relics": sorted(relics),
                "gold": gold,
            })

            if outcome == "lose":
                # Assign values: run died here
                _assign_run_values(combat_samples_by_floor, floor_reached,
                                   len(room_sequence), 0, max_hp,
                                   deck_change_samples, option_samples,
                                   combat_hp_data=combat_hp_data,
                                   boss_floors=boss_floors,
                                   outcome="lose")
                _arch = classify_deck(deck)
                return FullRunResult(
                    outcome="lose", floor_reached=floor_reached,
                    final_hp=0, max_hp=max_hp,
                    combats_won=combats_won, combats_fought=combats_fought,
                    deck_size=len(deck), samples=all_samples,
                    deck_samples=deck_change_samples,
                    option_samples=option_samples, combat_log=combat_log,
                    run_log=run_log,
                    archetype=_arch.archetype,
                    archetype_commitment=_arch.commitment,
                    boss_detail=boss_detail_holder,
                    final_deck=[c.id for c in deck],
                    final_relics=sorted(relics),
                )

            combats_won += 1

            # End-of-combat relic effects (healing etc.)
            # Build a temporary state to apply relic effects
            _post_player = PlayerState(hp=hp_after, max_hp=max_hp, energy=0, max_energy=0)
            _post_state = CombatState(player=_post_player, enemies=[], relics=frozenset(relics))
            end_combat_relics(_post_state)
            hp = _post_state.player.hp

            # Post-combat: gold, potions, card/relic rewards
            gold_range = GOLD_REWARDS.get(room_type, (10, 20))
            gold += rng.randint(*gold_range)

            if rng.random() < POTION_DROP_CHANCE and len(potions) < potion_slots_cap:
                pot = _random_potion(rng)
                potions.append(dict(pot))

            # Elite relic drop — V7: deck-aware 1-of-3 pick from the
            # implemented-effects pool.  Previously this was a uniform
            # rng.choice over the same pool which ignored deck shape.
            if room_type == "elite":
                granted, offered = _pick_best_relic(
                    IMPLEMENTED_RELIC_POOL, deck, relics, rng,
                )
                if granted is not None:
                    hp, max_hp, gold, potion_slots_cap = _grant_relic(
                        relics, granted,
                        hp=hp, max_hp=max_hp, gold=gold,
                        potions=potions, potion_slots=potion_slots_cap,
                    )
                    # Apply OOC relic effects
                    hp, max_hp = _apply_relic_ooc_effects(
                        granted, deck, card_db, hp, max_hp)
                    run_log.append({
                        "floor": floor_num, "type": "relic_reward",
                        "source": "elite", "relic": granted,
                        "offered": offered,
                        "deck_size": len(deck),
                        "deck_cards": [c.id for c in deck],
                        "hp": hp,
                        "max_hp": max_hp,
                    })

            if room_type != "boss":
                offered = _offer_card_rewards(pools, deck)
                pick, deck_sample = _network_pick_card(
                    offered, deck, hp, max_hp, floor_num,
                    mcts, vocabs, config, card_db,
                    relics=frozenset(relics),
                )

                # ----------------------------------------------------------
                # Decaying exploration for underpicked cards (Option D).
                # If the network picked a card, sometimes override to an
                # alternative that has high empirical win correlation but
                # few picks so far. Exploration rate decays as the card
                # accumulates picks: rate = 0.3 / sqrt(1 + times_picked).
                # At 0 picks: 30%, at 4 picks: 13%, at 9 picks: 10%,
                # at 25 picks: 6%, at 100 picks: 3%.
                # ----------------------------------------------------------
                _EXPLORE_CARD_LIFT = {
                    "BLADE_DANCE": 1.46, "HIDDEN_DAGGERS": 1.37,
                    "DAGGER_THROW": 1.35, "LEADING_STRIKE": 1.38,
                    "PANIC_BUTTON": 1.44, "CLOAK_AND_DAGGER": 1.22,
                    "SUCKER_PUNCH": 1.66, "DARK_SHACKLES": 2.4,
                    "BOUNCING_FLASK": 1.73, "EXPOSE": 1.67,
                    "PRECISE_CUT": 1.58, "BACKSTAB": 1.37,
                }
                if pick is not None and offered and rng.random() < 0.3:
                    # Find underexplored high-lift alternatives
                    explore_candidates = []
                    for i, card in enumerate(offered):
                        if card.id == (pick.id if pick else ""):
                            continue
                        lift = _EXPLORE_CARD_LIFT.get(card.id, 0)
                        if lift >= 1.2:
                            picks_so_far = _card_explore_counts.get(card.id, 0)
                            explore_rate = 0.3 / (1 + picks_so_far) ** 0.5
                            if rng.random() < explore_rate:
                                explore_candidates.append((card, i))
                    if explore_candidates:
                        explore_card, explore_idx = rng.choice(explore_candidates)
                        pick = explore_card
                        if deck_sample is not None:
                            deck_sample.chosen_idx = explore_idx

                # Track picks for exploration decay
                if pick is not None:
                    _card_explore_counts[pick.id] = _card_explore_counts.get(pick.id, 0) + 1

                run_log.append({
                    "floor": floor_num, "type": "card_reward",
                    "offered": [c.id for c in offered],
                    "picked": pick.id if pick else None,
                    "skipped": pick is None,
                })
                if pick:
                    # Apply egg auto-upgrades
                    if "FROZEN_EGG" in relics and pick.card_type == CardType.POWER:
                        _upgrade_card(pick, card_db)
                    elif "MOLTEN_EGG" in relics and pick.card_type == CardType.ATTACK:
                        _upgrade_card(pick, card_db)
                    elif "TOXIC_EGG" in relics and pick.card_type == CardType.SKILL:
                        _upgrade_card(pick, card_db)

                    # BING_BONG: add TWO copies instead of one
                    if "BING_BONG" in relics:
                        deck.append(pick)
                        deck.append(copy.copy(pick))
                    else:
                        deck.append(pick)
                if deck_sample:
                    deck_change_samples.append(deck_sample)

            if room_type == "boss":
                granted, offered = _pick_best_relic(
                    IMPLEMENTED_RELIC_POOL, deck, relics, rng,
                )
                if granted is not None:
                    hp, max_hp, gold, potion_slots_cap = _grant_relic(
                        relics, granted,
                        hp=hp, max_hp=max_hp, gold=gold,
                        potions=potions, potion_slots=potion_slots_cap,
                    )
                    # Apply OOC relic effects
                    hp, max_hp = _apply_relic_ooc_effects(
                        granted, deck, card_db, hp, max_hp)
                    run_log.append({
                        "floor": floor_num, "type": "relic_reward",
                        "source": "boss", "relic": granted,
                        "offered": offered,
                        "deck_size": len(deck),
                        "deck_cards": [c.id for c in deck],
                        "hp": hp,
                        "max_hp": max_hp,
                    })
                _assign_run_values(combat_samples_by_floor, floor_reached,
                                   len(room_sequence), hp, max_hp,
                                   deck_change_samples, option_samples,
                                   combat_hp_data=combat_hp_data,
                                   boss_floors=boss_floors,
                                   outcome="win")
                _arch = classify_deck(deck)
                return FullRunResult(
                    outcome="win", floor_reached=floor_reached,
                    final_hp=hp, max_hp=max_hp,
                    combats_won=combats_won, combats_fought=combats_fought,
                    deck_size=len(deck), samples=all_samples,
                    deck_samples=deck_change_samples,
                    option_samples=option_samples, combat_log=combat_log,
                    run_log=run_log,
                    archetype=_arch.archetype,
                    archetype_commitment=_arch.commitment,
                    boss_detail=boss_detail_holder,
                    final_deck=[c.id for c in deck],
                    final_relics=sorted(relics),
                )

        elif room_type == "rest":
            # Network-scored rest site decision
            try:
                network = mcts.network
                player = PlayerState(hp=hp, max_hp=max_hp, energy=3, max_energy=3,
                                     draw_pile=list(deck))
                dummy = CombatState(player=player, enemies=[], floor=floor_num, gold=gold, relics=frozenset(relics))
                st = encode_state(dummy, vocabs, config)
                st = {k: v.to(mcts.device) for k, v in st.items()}

                opt_types = [OPTION_REST]
                opt_cards = [0]
                deck_indices = [None]  # maps option idx → deck idx

                for di, card in enumerate(deck):
                    if not card.upgraded and card.card_type not in ("Status", "Curse"):
                        up = card_db.get_upgraded(card.id)
                        if up:
                            opt_types.append(OPTION_SMITH)
                            opt_cards.append(vocabs.cards.get(card.id.rstrip("+")))
                            deck_indices.append(di)

                with torch.no_grad():
                    hidden = network.encode_state(**st)
                    best_idx, scores = network.pick_best_option(
                        hidden, opt_types, opt_cards)

                shadow_idx = _shadow.shadow_pick_rest(
                    opt_types, deck_indices, deck, hp, max_hp, floor_num,
                    relics=frozenset(relics),
                )

                # ----------------------------------------------------------
                # Approach 3: HP-aware exploration forcing for REST
                # Override the network to REST when the shadow recommends it.
                # Rate scales with how damaged the player is:
                #   - Below 40% HP: 40% chance (critical — need healing data)
                #   - 40-60% HP:    25% chance (gray zone — need both outcomes)
                #   - Above 60% HP: 10% chance (light exploration)
                # This ensures the training buffer has episodes where the
                # agent healed at different HP thresholds, so the value
                # signal teaches correct rest-vs-smith tradeoffs.
                # ----------------------------------------------------------
                hp_ratio = hp / max(1, max_hp)
                if hp_ratio < 0.4:
                    rest_explore_rate = 0.40
                elif hp_ratio < 0.6:
                    rest_explore_rate = 0.25
                else:
                    rest_explore_rate = 0.10
                if (best_idx != 0
                        and shadow_idx == 0
                        and rng.random() < rest_explore_rate):
                    best_idx = 0  # force REST for exploration

                option_samples.append(OptionSample(
                    state_tensors={k: v.cpu() for k, v in st.items()},
                    option_types=opt_types, option_cards=opt_cards,
                    chosen_idx=best_idx, value=0.0,
                    shadow_chosen_idx=shadow_idx,
                ))

                if best_idx == 0:
                    hp = min(hp + int(max_hp * 0.3), max_hp)
                    # Apply tea set bonus for next combat
                    if "VENERABLE_TEA_SET" in relics:
                        _tea_set_bonus = 2
                    elif "FAKE_VENERABLE_TEA_SET" in relics:
                        _tea_set_bonus = 1
                    run_log.append({
                        "floor": floor_num, "type": "rest",
                        "action": "rest", "hp_before": hp - int(max_hp * 0.3),
                        "hp_after": hp, "max_hp": max_hp,
                    })
                else:
                    di = deck_indices[best_idx]
                    _smithed_card = None
                    if di is not None and di < len(deck):
                        _smithed_card = deck[di].id
                        upgraded = card_db.get_upgraded(deck[di].id)
                        if upgraded:
                            deck[di] = copy.copy(upgraded)
                    run_log.append({
                        "floor": floor_num, "type": "rest",
                        "action": "smith", "card_upgraded": _smithed_card,
                        "hp": hp, "max_hp": max_hp,
                    })
            except Exception:
                # Fallback to heuristic
                decision = _rest_site_decision(hp, max_hp, deck, card_db, rng)
                if decision["action"] == "rest":
                    hp_before_rest = hp
                    hp = min(hp + decision["hp_delta"], max_hp)
                    # Apply tea set bonus for next combat
                    if "VENERABLE_TEA_SET" in relics:
                        _tea_set_bonus = 2
                    elif "FAKE_VENERABLE_TEA_SET" in relics:
                        _tea_set_bonus = 1
                    run_log.append({
                        "floor": floor_num, "type": "rest",
                        "action": "rest", "hp_before": hp_before_rest,
                        "hp_after": hp, "max_hp": max_hp,
                    })
                else:
                    idx = decision["upgrade_card_idx"]
                    _smithed_card = deck[idx].id if idx is not None and idx < len(deck) else None
                    if idx is not None and idx < len(deck):
                        upgraded = card_db.get_upgraded(deck[idx].id)
                        if upgraded:
                            deck[idx] = copy.copy(upgraded)
                    run_log.append({
                        "floor": floor_num, "type": "rest",
                        "action": "smith", "card_upgraded": _smithed_card,
                        "hp": hp, "max_hp": max_hp,
                    })

        elif room_type == "event":
            if event_idx < len(events_list):
                eid = events_list[event_idx]
                event_idx += 1
            else:
                eid = rng.choice(events_list) if events_list else None
            if eid:
                # IMPROVEMENTS.md #4: network picks the event choice.
                # Enumerate all options with their resolved effect tuples,
                # forward-pass the option head, and apply the chosen one.
                # Build an OptionSample so the run outcome back-propagates
                # into this decision via _assign_run_values.
                from ..simulator import enumerate_event_options
                from .self_play import OPTION_EVENT_CHOICE

                choices = enumerate_event_options(
                    eid, deck, hp, max_hp, gold, card_db, rng)
                chosen_changes: dict | None = None
                if choices:
                    try:
                        network = mcts.network
                        player = PlayerState(
                            hp=hp, max_hp=max_hp,
                            energy=3, max_energy=3,
                            draw_pile=list(deck),
                        )
                        dummy = CombatState(
                            player=player, enemies=[],
                            floor=floor_num, gold=gold,
                            relics=frozenset(relics),
                        )
                        st = encode_state(dummy, vocabs, config)
                        st = {k: v.to(mcts.device) for k, v in st.items()}

                        opt_types = [OPTION_EVENT_CHOICE] * len(choices)
                        # V10: use real EVENT_CHOICE_VOCAB ids — each
                        # (event_id, option_idx) pair maps to a unique
                        # slot in the dedicated event_choice_embed table.
                        opt_cards = [c["vocab_id"] for c in choices]

                        with torch.no_grad():
                            hidden = network.encode_state(**st)
                            best_idx, _scores = network.pick_best_option(
                                hidden, opt_types, opt_cards)

                        shadow_idx = _shadow.shadow_pick_event(
                            eid, hp, max_hp, gold, deck)

                        option_samples.append(OptionSample(
                            state_tensors={k: v.cpu() for k, v in st.items()},
                            option_types=opt_types,
                            option_cards=opt_cards,
                            chosen_idx=best_idx, value=0.0,
                            shadow_chosen_idx=shadow_idx,
                        ))
                        chosen_changes = choices[best_idx]["changes"]
                    except Exception:
                        # Fallback to heuristic pick on any network error
                        shadow_idx = _shadow.shadow_pick_event(
                            eid, hp, max_hp, gold, deck)
                        chosen_changes = choices[shadow_idx]["changes"]

                # Apply effects atomically (None → no-op)
                changes = chosen_changes or _simulate_event(
                    eid, deck, hp, max_hp, gold, card_db, rng)
                hp = max(1, min(hp + changes["hp_delta"], max_hp + changes["max_hp_delta"]))
                max_hp += changes["max_hp_delta"]
                gold = max(0, gold + changes["gold_delta"])
                for idx in sorted(changes["cards_removed"], reverse=True):
                    if idx < len(deck):
                        deck.pop(idx)
                for card in changes["cards_added"]:
                    # Apply egg auto-upgrades
                    if "FROZEN_EGG" in relics and card.card_type == CardType.POWER:
                        _upgrade_card(card, card_db)
                    elif "MOLTEN_EGG" in relics and card.card_type == CardType.ATTACK:
                        _upgrade_card(card, card_db)
                    elif "TOXIC_EGG" in relics and card.card_type == CardType.SKILL:
                        _upgrade_card(card, card_db)

                    # BING_BONG: add TWO copies instead of one
                    if "BING_BONG" in relics:
                        deck.append(card)
                        deck.append(copy.copy(card))
                    else:
                        deck.append(card)
                # V7: relic reward if event option grants a relic.
                _event_relic = None
                _event_relic_offered = []
                if changes.get("grants_relic"):
                    granted, offered = _pick_best_relic(
                        IMPLEMENTED_RELIC_POOL, deck, relics, rng,
                    )
                    if granted is not None:
                        _event_relic = granted
                        _event_relic_offered = offered
                        hp, max_hp, gold, potion_slots_cap = _grant_relic(
                            relics, granted,
                            hp=hp, max_hp=max_hp, gold=gold,
                            potions=potions, potion_slots=potion_slots_cap,
                        )
                        # Apply OOC relic effects
                        hp, max_hp = _apply_relic_ooc_effects(
                            granted, deck, card_db, hp, max_hp)

                run_log.append({
                    "floor": floor_num, "type": "event",
                    "event_id": eid,
                    "hp_delta": changes.get("hp_delta", 0),
                    "max_hp_delta": changes.get("max_hp_delta", 0),
                    "gold_delta": changes.get("gold_delta", 0),
                    "cards_removed": changes.get("cards_removed", []),
                    "cards_added": [c.id if hasattr(c, 'id') else str(c) for c in changes.get("cards_added", [])],
                    "relic_granted": _event_relic,
                    "relic_offered": _event_relic_offered,
                    "hp_after": hp, "max_hp": max_hp, "gold": gold,
                })

        elif room_type == "treasure":
            granted, offered = _pick_best_relic(
                IMPLEMENTED_RELIC_POOL, deck, relics, rng,
            )
            if granted is not None:
                hp, max_hp, gold, potion_slots_cap = _grant_relic(
                    relics, granted,
                    hp=hp, max_hp=max_hp, gold=gold,
                    potions=potions, potion_slots=potion_slots_cap,
                )
                # Apply OOC relic effects
                hp, max_hp = _apply_relic_ooc_effects(
                    granted, deck, card_db, hp, max_hp)
                run_log.append({
                    "floor": floor_num, "type": "treasure",
                    "relic": granted,
                    "offered": offered,
                    "deck_size": len(deck),
                    "deck_cards": [c.id for c in deck],
                    "hp": hp,
                    "max_hp": max_hp,
                })

        elif room_type == "shop":
            # Network-driven multi-step shop
            try:
                network = mcts.network
                # V10: bumped from 3 → 6 to match STS2's real shop size
                # (parity fix from docs/shop_parity.md #17).
                shop_cards = _offer_card_rewards(pools, deck, 6)
                shop_costs = []
                for sc in shop_cards:
                    cost = 75
                    for rarity, pool_cards in pools.items():
                        if any(c.id == sc.id for c in pool_cards):
                            cost = SHOP_CARD_COSTS.get(rarity, 75)
                            break
                    shop_costs.append(cost)

                # Offer 2 random potions at the shop
                shop_potions = [_random_potion(rng) for _ in range(2)]

                # Offer up to 3 relics (parity fix from docs/shop_parity.md #18).
                # Filter out already-owned relics, same as _simulate_shop.
                shop_relic_pool = _get_shop_relic_pool()
                eligible_relics = [r for r in shop_relic_pool if r not in relics]
                shop_relics: list[str | None] = []
                if eligible_relics:
                    shop_relics = list(rng.sample(
                        eligible_relics, min(3, len(eligible_relics))))

                _shop_actions = []  # track for run_log
                for _step in range(6):
                    player = PlayerState(hp=hp, max_hp=max_hp, energy=3,
                                         max_energy=3, draw_pile=list(deck),
                                         potions=[dict(p) for p in potions])
                    dummy = CombatState(player=player, enemies=[],
                                        floor=floor_num, gold=gold,
                                        relics=frozenset(relics))
                    st = encode_state(dummy, vocabs, config)
                    st = {k: v.to(mcts.device) for k, v in st.items()}

                    opt_types = []
                    opt_cards = []
                    actions = []  # ("remove", di) | ("buy", si, cost) | ("potion", pi) | ("relic", ri) | ("leave",)

                    # Remove options (Strike/Defend only)
                    if gold >= SHOP_CARD_REMOVE_COST:
                        for di, card in enumerate(deck):
                            if card.name in ("Strike", "Defend") and not card.upgraded:
                                opt_types.append(OPTION_SHOP_REMOVE)
                                opt_cards.append(vocabs.cards.get(card.id.rstrip("+")))
                                actions.append(("remove", di))

                    # Buy relic options — relics go before cards because
                    # they're the highest-impact shop decision (see
                    # _simulate_shop's priority order and boss-log data).
                    if gold >= SHOP_RELIC_COST:
                        for ri, rname in enumerate(shop_relics):
                            if rname is not None:
                                opt_types.append(OPTION_SHOP_BUY_RELIC)
                                opt_cards.append(
                                    vocabs.relics.get(rname) or 0)
                                actions.append(("relic", ri))

                    # Buy card options
                    for si, (sc, cost) in enumerate(zip(shop_cards, shop_costs)):
                        if sc is not None and gold >= cost:
                            opt_types.append(OPTION_SHOP_BUY)
                            opt_cards.append(vocabs.cards.get(sc.id.rstrip("+")))
                            actions.append(("buy", si, cost))

                    # Buy potion options (if we have room and gold)
                    if gold >= SHOP_POTION_COST and len(potions) < POTION_SLOTS:
                        for pi, pot in enumerate(shop_potions):
                            if pot is not None:
                                opt_types.append(OPTION_SHOP_BUY_POTION)
                                opt_cards.append(0)  # Potions aren't cards
                                actions.append(("potion", pi))

                    # Leave option (always available)
                    opt_types.append(OPTION_SHOP_LEAVE)
                    opt_cards.append(0)
                    actions.append(("leave",))

                    if len(opt_types) == 1:
                        break  # only leave available

                    with torch.no_grad():
                        hidden = network.encode_state(**st)
                        best_idx, scores = network.pick_best_option(
                            hidden, opt_types, opt_cards)

                    shadow_idx = _shadow.shadow_pick_shop(
                        opt_types, actions, deck, hp, max_hp, gold,
                        floor_num, relics=frozenset(relics),
                    )
                    option_samples.append(OptionSample(
                        state_tensors={k: v.cpu() for k, v in st.items()},
                        option_types=opt_types, option_cards=opt_cards,
                        chosen_idx=best_idx, value=0.0,
                        shadow_chosen_idx=shadow_idx,
                    ))

                    action = actions[best_idx]
                    if action[0] == "leave":
                        break
                    elif action[0] == "remove":
                        _shop_actions.append({"action": "remove", "card": deck[action[1]].id, "cost": SHOP_CARD_REMOVE_COST})
                        deck.pop(action[1])
                        gold -= SHOP_CARD_REMOVE_COST
                    elif action[0] == "relic":
                        rname = shop_relics[action[1]]
                        shop_relics[action[1]] = None  # sold out
                        if rname:
                            _shop_actions.append({"action": "buy_relic", "relic": rname, "cost": SHOP_RELIC_COST})
                            hp, max_hp, gold_after, potion_slots_cap = _grant_relic(
                                relics, rname,
                                hp=hp, max_hp=max_hp, gold=gold,
                                potions=potions, potion_slots=potion_slots_cap,
                            )
                            # Apply OOC relic effects
                            hp, max_hp = _apply_relic_ooc_effects(
                                rname, deck, card_db, hp, max_hp)
                            gold = gold_after - SHOP_RELIC_COST
                    elif action[0] == "buy":
                        card_to_buy = shop_cards[action[1]]
                        _shop_actions.append({"action": "buy_card", "card": card_to_buy.id, "cost": action[2]})
                        # Apply egg auto-upgrades
                        if "FROZEN_EGG" in relics and card_to_buy.card_type == CardType.POWER:
                            _upgrade_card(card_to_buy, card_db)
                        elif "MOLTEN_EGG" in relics and card_to_buy.card_type == CardType.ATTACK:
                            _upgrade_card(card_to_buy, card_db)
                        elif "TOXIC_EGG" in relics and card_to_buy.card_type == CardType.SKILL:
                            _upgrade_card(card_to_buy, card_db)

                        # BING_BONG: add TWO copies instead of one
                        if "BING_BONG" in relics:
                            deck.append(card_to_buy)
                            deck.append(copy.copy(card_to_buy))
                        else:
                            deck.append(card_to_buy)
                        gold -= action[2]
                        shop_cards[action[1]] = None  # sold out
                    elif action[0] == "potion":
                        _shop_actions.append({"action": "buy_potion", "potion": shop_potions[action[1]].get("name", "?"), "cost": SHOP_POTION_COST})
                        potions.append(dict(shop_potions[action[1]]))
                        gold -= SHOP_POTION_COST
                        shop_potions[action[1]] = None  # sold out

                run_log.append({
                    "floor": floor_num, "type": "shop",
                    "gold_before": gold + sum(a.get("cost", 0) for a in _shop_actions),
                    "gold_after": gold,
                    "actions": _shop_actions,
                })

            except Exception:
                # Fallback to heuristic
                shop_result = _simulate_shop(
                    deck, gold, card_db, pools, rng,
                    relics=frozenset(relics))
                gold += shop_result["gold_delta"]
                for idx in sorted(shop_result.get("cards_removed", []), reverse=True):
                    if idx < len(deck):
                        deck.pop(idx)
                for card in shop_result.get("cards_added", []):
                    # Apply egg auto-upgrades
                    if "FROZEN_EGG" in relics and card.card_type == CardType.POWER:
                        _upgrade_card(card, card_db)
                    elif "MOLTEN_EGG" in relics and card.card_type == CardType.ATTACK:
                        _upgrade_card(card, card_db)
                    elif "TOXIC_EGG" in relics and card.card_type == CardType.SKILL:
                        _upgrade_card(card, card_db)

                    # BING_BONG: add TWO copies instead of one
                    if "BING_BONG" in relics:
                        deck.append(card)
                        deck.append(copy.copy(card))
                    else:
                        deck.append(card)
                for relic_name in shop_result.get("relics_bought", []):
                    hp, max_hp, gold, potion_slots_cap = _grant_relic(
                        relics, relic_name,
                        hp=hp, max_hp=max_hp, gold=gold,
                        potions=potions, potion_slots=potion_slots_cap,
                    )
                    # Apply OOC relic effects
                    hp, max_hp = _apply_relic_ooc_effects(
                        relic_name, deck, card_db, hp, max_hp)
                run_log.append({
                    "floor": floor_num, "type": "shop",
                    "gold_delta": shop_result.get("gold_delta", 0),
                    "actions": [{"action": "heuristic_fallback"}],
                })

    # Completed all floors without boss (shouldn't happen normally)
    _assign_run_values(combat_samples_by_floor, floor_reached,
                       len(room_sequence), hp, max_hp,
                       deck_change_samples, option_samples,
                       combat_hp_data=combat_hp_data,
                       boss_floors=boss_floors,
                       outcome="lose")
    _arch = classify_deck(deck)
    return FullRunResult(
        outcome="lose", floor_reached=floor_reached,
        final_hp=hp, max_hp=max_hp,
        combats_won=combats_won, combats_fought=combats_fought,
        deck_size=len(deck), samples=all_samples,
        deck_samples=deck_change_samples,
        option_samples=option_samples, combat_log=combat_log,
        run_log=run_log,
        archetype=_arch.archetype,
        archetype_commitment=_arch.commitment,
        boss_detail=boss_detail_holder,
        final_deck=[c.id for c in deck],
        final_relics=sorted(relics),
    )


def _assign_run_values(
    combat_samples_by_floor: dict[int, list[TrainingSample]],
    floor_reached: int,
    total_floors: int,
    final_hp: int,
    max_hp: int,
    deck_change_samples: list | None = None,
    option_samples: list | None = None,
    combat_hp_data: dict[int, tuple[int, int, int]] | None = None,
    boss_floors: set[int] | None = None,
    outcome: str = "lose",
) -> None:
    """Assign training values blending per-combat HP conservation with run outcome.

    Each combat gets a dense local signal based on how efficiently it was played
    (HP retained, potions conserved), blended with the sparse run-level outcome.
    This teaches the network that winning a combat at 5 HP is worse than at 40 HP.

    Boss fights are treated differently: HP conservation doesn't matter (HP resets
    next act), only winning counts. Potion use is barely penalised because losing
    is much worse than dumping consumables.

    Run-level scoring (2026-04-10 rebalance):
    - Beating the boss is worth a flat +1.0 regardless of final HP — winning
      is the only thing that matters. This creates a large cliff between
      "died at floor 16" and "beat the boss" so the network valorises wins
      over incremental progress.
    - Losing is scored on a compressed [-0.5, ~+0.1] scale based on how far
      the run got, keeping enough signal to teach "floor 12 is better than
      floor 3" without letting good-but-losing runs look comparable to wins.
    """
    # --- Run-level value (sparse, based on overall outcome) ---
    # Outcome is always passed explicitly by play_full_run's three call
    # sites; the equality check is authoritative.
    won = (outcome == "win")
    if won:
        # Flat maximum reward for beating the boss. HP doesn't matter —
        # 3/70 and 70/70 are the same story: "you won Act 1".
        run_value = 1.0
    else:
        # Losses: scale by floor reached. Max loss-value is ~+0.1 at
        # boss-floor death, min is ~-0.5 at floor-1 death. HP on loss
        # gets a small bonus (surviving longer at each floor is still
        # informative), but the floor weight dominates.
        base = floor_reached / max(1, total_floors) * 0.6
        hp_bonus = final_hp / max(1, max_hp) * 0.1
        run_value = base + hp_bonus - 0.5  # [~-0.5, ~+0.2]
    run_value = max(-1.0, min(1.0, run_value))

    # --- Per-combat values (dense, based on HP conservation) ---
    if combat_hp_data is None:
        combat_hp_data = {}
    if boss_floors is None:
        boss_floors = set()

    discount = 0.95       # run-level: earlier combats get less certain values
    turn_discount = 0.99  # within-combat temporal discount
    sorted_floors = sorted(combat_samples_by_floor.keys(), reverse=True)

    for i, floor in enumerate(sorted_floors):
        # Run-level contribution (discounted by distance from end)
        run_component = run_value * (discount ** i)

        is_boss = floor in boss_floors

        if is_boss:
            # Boss fights: winning is everything. The loss penalty is
            # scaled by entering HP — arriving at the boss crippled (say
            # 15/80 HP) means the loss was mostly baked in by prior Act 1
            # combats, not by boss play. Teaching the network "you lost
            # with -1.0 regardless of how you got there" adds noise.
            #
            # Formula: full-HP loss = -1.0, zero-HP loss = -0.3.
            # A floor of -0.3 keeps some loss signal even when broken.
            if floor in combat_hp_data:
                hp_before, hp_after, potions_used = combat_hp_data[floor]
                if hp_after <= 0:
                    entry_ratio = max(0.0, min(1.0, hp_before / max(1, max_hp)))
                    combat_value = -(0.3 + 0.7 * entry_ratio)
                else:
                    # Won the boss fight — strong positive signal.
                    # No potion penalty: using potions at the boss is the
                    # correct strategy (they're pre-dumped for burst damage).
                    combat_value = 1.0
            else:
                combat_value = 0.0

            # Boss: weight toward win/lose outcome, less run-level blend
            blended = 0.7 * combat_value + 0.3 * run_component
        else:
            # Non-boss: HP conservation matters for surviving the run.
            # Potion use is a small nudge (0.03 per potion, down from
            # 0.10) — we'd rather the network cashes potions to survive
            # than hoards them into a loss.
            if floor in combat_hp_data:
                hp_before, hp_after, potions_used = combat_hp_data[floor]
                if hp_before <= 0:
                    combat_value = -1.0
                else:
                    hp_retained = hp_after / max(1, hp_before)
                    damage_fraction = (hp_before - hp_after) / max(1, max_hp)
                    potion_penalty = potions_used * 0.03
                    # Cross-fight HP preservation bonus: reward ending
                    # fights with high absolute HP so the agent learns
                    # to manage health as a multi-fight resource.
                    # +0.15 at full HP, 0 at half, -0.15 at near-death.
                    hp_abs_ratio = hp_after / max(1, max_hp)
                    hp_preservation_bonus = (hp_abs_ratio - 0.5) * 0.3
                    combat_value = (hp_retained
                                    - damage_fraction * 0.5
                                    - potion_penalty
                                    + hp_preservation_bonus)
                    combat_value = max(-1.0, min(1.0, combat_value))
            else:
                combat_value = 0.0

            blended = 0.5 * combat_value + 0.5 * run_component

        floor_samples = combat_samples_by_floor[floor]
        n = len(floor_samples)
        for j, sample in enumerate(floor_samples):
            turns_from_end = n - 1 - j
            sample.value = blended * (turn_discount ** turns_from_end)
            # Per-step penalties: wasted energy (0.15) + any block penalty
            penalty = sample.value_penalty
            if sample.wasted_energy:
                penalty += 0.15
            if penalty > 0:
                sample.value = max(-1.0, sample.value - penalty)

    # Deck change and option samples get a run value with HP preservation bonus.
    #
    # The base run_value is binary-ish (+1 win, scaled loss). But for NON-combat
    # decisions (rest/map/shop/card picks), we want the agent to learn that
    # arriving at the boss with more HP is better. A rest-site heal that lets
    # you enter the boss at 55/70 HP instead of 25/70 should score higher even
    # if both runs ultimately lost.
    #
    # HP preservation bonus: for runs that reached the boss floor (floor >= 16),
    # add up to +0.15 for arriving with full HP, 0 at half HP, -0.15 at low HP.
    # This is on top of run_value, so a loss-at-boss with good HP management
    # gets a less negative value than a loss-at-boss with poor HP management.
    hp_preservation = 0.0
    if floor_reached >= total_floors - 1 and max_hp > 0:
        # Boss was reached — reward HP at boss entry
        boss_floor = max(sorted_floors) if sorted_floors else floor_reached
        if boss_floor in combat_hp_data:
            boss_entry_hp = combat_hp_data[boss_floor][0]  # hp_before boss
        else:
            boss_entry_hp = final_hp
        hp_ratio = boss_entry_hp / max(1, max_hp)
        hp_preservation = (hp_ratio - 0.5) * 0.3  # [-0.15, +0.15]

    option_value = max(-1.0, min(1.0, run_value + hp_preservation))

    if deck_change_samples:
        for sample in deck_change_samples:
            sample.value = option_value
    if option_samples:
        for sample in option_samples:
            sample.value = option_value
