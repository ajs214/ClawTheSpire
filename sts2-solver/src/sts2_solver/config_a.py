"""Config Profile A — "Self-Play" (lean on the network where training has signal).

Experiment design (2026-04-11):
  Profile A leans on the trained network as much as is practical now.
  Profile B is the deterministic baseline that routes the same screens
  through the rule-based advisor. Both profiles carry IDENTICAL scalar
  weights — EVALUATOR and STRATEGY values are midpoints between the
  old Champion (A) and Challenger (B) tunings — so the only experimental
  variable is ROUTING, not numerics. This lets an A/B run cleanly test
  "does the network beat the deterministic advisor on map/rest/shop?"
  without confounding the result with weight changes.

  Profile A sets USE_NETWORK_ROUTING = True:
    - Map, rest, and shop defer to the network's option head, falling
      back to deterministic only when the network returns None (error
      or no confident match).
    - Combat is always MCTS (same in both profiles).

  Profile B sets USE_NETWORK_ROUTING = False:
    - Map, rest, shop all route directly to the deterministic advisor
      with no network attempt.
    - Combat is still MCTS (the network's combat head is not part of
      the experiment — we know it beats deterministic combat).

  Universal network routes (both profiles, NOT gated by USE_NETWORK_ROUTING):
    - Card reward: _az_decide_card_reward → network option head picks
      take-card-N / skip. Falls back to decide_card_reward (organic
      picker) only on network failure.
    - Act 1+ events: _az_decide_event_choice → network option head
      picks among unlocked event options. Falls back to
      decide_event_default (sim scorer) on network failure.
    - Neow events (profile A only for now): _az_decide_neow → network
      picks a blessing, tag-matched back to a live option. Profile B
      uses the deterministic Neow keyword scorer.

Run: bash play.sh --profile a   (defaults; or set STS2_CONFIG_PROFILE=a)
Compare: bash play.sh --profile b

===========================================================================
DECISION ROUTING — who actually drives each screen (verified 2026-04-11)
===========================================================================

| Screen          | Training (full_run)        | Live play (runner, A=Self-Play / B=Deterministic)     | Learn signal? |
|-----------------|----------------------------|-------------------------------------------------------|---------------|
| Combat          | solve_turn (heuristic DFS) | AlphaZero MCTS (both profiles)                        | YES (MCTS)    |
| Map node        | network.pick_best_option   | A: network.pick_best_option → deterministic fallback  | YES (value)   |
|                 |                            | B: decide_map (deterministic_advisor)                 |               |
| Rest site       | network.pick_best_option   | A: network.pick_best_option → deterministic fallback  | YES (value)   |
|                 |                            | B: decide_rest (deterministic_advisor)                |               |
| Shop            | network.pick_best_option   | A: network.pick_best_option → deterministic fallback  | YES (value)   |
|                 |                            | B: decide_shop (deterministic_advisor)                |               |
| Card reward     | network.pick_best_option   | network.pick_best_option → organic_picker fallback    | YES (value)   |
|                 |   (OPTION_CARD_REWARD/SKIP)|   (BOTH profiles — not gated by USE_NETWORK_ROUTING)  |               |
| Event (Act 1+)  | network.pick_best_option   | network.pick_best_option → sim scorer fallback        | YES (value)   |
|                 |   (OPTION_EVENT_CHOICE)    |   (BOTH profiles — not gated by USE_NETWORK_ROUTING)  |               |
| Event (Neow)    | network option head        | A: _az_decide_neow → decide_neow fallback             | YES (value)   |
|                 |   + decide_neow fallback   | B: decide_neow (keyword rule)                         |               |
| Boss relic      | _pick_best_relic (rule)    | decide_boss_relic (rule, both profiles)               | NO — rule     |
| Elite relic     | _pick_best_relic (rule)    | deterministic_advisor (rule, both profiles)           | NO — rule     |
| Treasure relic  | _pick_best_relic (rule)    | deterministic_advisor (rule, both profiles)           | NO — rule     |
| Neow bonus      | not modeled                | runner auto-handler (both profiles)                   | NO — absent   |
| Capstone/bundle | not modeled                | runner auto-handler (both profiles)                   | NO — absent   |

KEY IMPLICATIONS:
- The ONLY live-play screens where A and B differ are: map, rest, shop,
  and Neow. Card reward and Act 1+ events are universally network-driven.
- The A/B test is strictly a routing experiment on map/rest/shop/Neow.
  Scalar EVALUATOR and STRATEGY values are identical between A and B
  (midpoints of the old Champion and Challenger tunings).
- STRATEGY scalars are mostly vestigial in Profile A (network handles
  the screens they used to gate), but are still consulted in
  deterministic-fallback paths. Do not delete them.
- CARD_TIERS in Profile A is now largely vestigial — the network
  handles card rewards in normal operation. Tiers only matter when the
  network card-reward handler fails and falls through to
  decide_card_reward's tier-list fallback path. Profile B still drives
  tier decisions via the organic picker.

See IMPROVEMENTS.md at the repo root for the full gap list and fixes.
===========================================================================
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Routing flag — the core A/B experimental variable.
# ---------------------------------------------------------------------------
# True  → live play tries the network's option head first for rest/map/shop
#         (and card reward once wired), falling back to deterministic only
#         when the network returns None. This is the "Self-Play" profile.
# False → live play skips the network entirely for those screens and goes
#         straight to the deterministic advisor. Combat still uses MCTS
#         in both profiles — that's not part of the experiment.
# Consumed by runner.py's non-combat screen dispatcher.
USE_NETWORK_ROUTING: bool = True


# Enemies that respawn or split on death (e.g. medium slimes → 2 small slimes).
# Killing these doesn't remove threats, so kill_bonus should be suppressed.
RESPAWNING_ENEMIES: frozenset[str] = frozenset({
    "EYE_WITH_TEETH",
    "LEAF_SLIME_M",
    "TWIG_SLIME_M",
})

# ---------------------------------------------------------------------------
# Evaluator weights — combat state scoring (character-agnostic)
# ---------------------------------------------------------------------------

EVALUATOR = {
    # ── A/B midpoints (2026-04-11) ──
    # All scalars below are midpoints between the old Champion (A) and
    # Challenger "Survive & Scale" (B) tunings. Both config_a.py and
    # config_b.py carry IDENTICAL values — the A/B experiment is now
    # about routing, not weights.

    # Damage scoring
    "kill_bonus": 56.0,              # midpoint of 50 / 62
    "buff_kill_bonus": 90.0,         # midpoint of 85 / 95
    "strength_kill_bonus_per": 11.0, # midpoint of 10 / 12
    "damage_alive_weight": 3.25,     # midpoint of 3.0 / 3.5
    "damage_dead_weight": 0.2,       # unchanged
    "kill_proximity_weight": 11.0,   # midpoint of 10 / 12

    # Enemy threat prioritisation
    "threat_buff_intent": 0.7,       # midpoint of 0.6 / 0.8
    "threat_strength_per": 0.09,     # midpoint of 0.08 / 0.10
    "threat_attack_damage_per": 0.0125, # midpoint of 0.01 / 0.015
    "threat_max_hp_per": 0.001,      # unchanged
    "threat_status_intent": 0.3,     # unchanged
    "threat_debuff_intent": 0.2,     # unchanged

    # Block scoring
    "effective_block_weight": 2.15,  # midpoint of 2.0 / 2.3
    "wasted_block_penalty": 1.35,    # midpoint of 1.5 / 1.2
    "idle_block_weight": 0.125,      # midpoint of 0.10 / 0.15

    # HP-aware block scaling
    "hp_block_threshold": 65,        # midpoint of 60 / 70
    "hp_block_scale": 0.055,         # midpoint of 0.05 / 0.06

    # Unblocked damage
    "unblocked_damage_penalty": 1.3, # midpoint of 1.2 / 1.4
    "lethal_damage_penalty": 600.0,  # midpoint of 500 / 700

    # Self-damage
    "self_damage_weight": 0.8,       # unchanged

    # Debuffs on enemies
    "vulnerable_value": 4.0,         # midpoint of 3.5 / 4.5
    "weak_vs_attack_value": 3.0,     # midpoint of 2.5 / 3.5
    "weak_vs_other_value": 1.1,      # midpoint of 1.0 / 1.2

    # Player buffs
    "strength_gained_value": 15.0,   # unchanged
    "dexterity_gained_value": 6.5,   # midpoint of 5.0 / 8.0

    # Poison scoring (Silent)
    "poison_future_discount": 1.15,  # midpoint of 1.0 / 1.3

    # Energy efficiency
    "unspent_energy_penalty": 12.0,  # unchanged

    # Card draw
    "card_draw_value": 7.5,          # midpoint of 7.0 / 8.0

    # 2-ply enemy simulation
    "enemy_sim_discount": 0.325,     # midpoint of 0.30 / 0.35
}


# ---------------------------------------------------------------------------
# Per-character power values for the evaluator
# ---------------------------------------------------------------------------

POWER_VALUES: dict[str, dict[str, float]] = {
    "ironclad": {
        "Demon Form": 10.0,         # (was 8.0)
        "Barricade": 8.0,           # (was 6.0)
        "Feel No Pain": 8.0,        # (was 4.0) — key exhaust synergy
        "Dark Embrace": 8.0,        # (was 4.0) — key exhaust synergy
        "Metallicize": 5.0,
        "Corruption": 8.0,          # (was 5.0) — exhaust holy trinity
    },
    "silent": {
        "Accuracy": 10.0,           # Doubles Shiv damage — core scaling
        "Infinite Blades": 9.0,     # Guaranteed Shiv each turn
        "Noxious Fumes": 8.0,       # AoE poison every turn — strong scaling
        "Tools of the Trade": 8.0,  # Draw + discard each turn — engine
        "Footwork": 8.0,            # Dexterity scaling — core block power
        "Well-Laid Plans": 7.0,     # Retain best card — critical consistency
        "Serpent Form": 7.0,        # Strong damage output
        "Accelerant": 7.0,          # Poison multiplier
        "Master Planner": 6.0,      # Makes skills Sly
        "Afterimage": 5.0,          # Block on card play
        "Abrasive": 5.0,            # Block + Thorns from Sly
        "Envenom": 4.0,             # Poison on attack damage
    },
}


# ---------------------------------------------------------------------------
# Card tier lists — per character
# Used in advisor prompts to guide card reward decisions
# ---------------------------------------------------------------------------

CARD_TIERS: dict[str, dict[str, list[str]]] = {
    "ironclad": {
        # Based on Mobalytics Ironclad guide + sim experiments.
        # AoE (Thunderclap, Whirlwind) and multi-hit (Twin Strike, Thrash) are
        # more valuable than single-target burst. Offering is the best card in
        # the game. The "holy trinity" (Corruption + Dark Embrace + Feel No Pain)
        # enables the strongest late-game engine.
        # Cleaned April 2026: removed STS1 cards not in STS2 data.
        "S": [
            "Offering", "Demon Form", "Corruption", "Impervious",
            "Whirlwind", "Inflame", "Feel No Pain", "Dark Embrace",
        ],
        "A": [
            "Thunderclap", "Twin Strike", "Battle Trance", "Shrug It Off",
            "Burning Pact", "Flame Barrier", "Thrash",
            "Pommel Strike", "True Grit", "Barricade", "Rupture",
            "Hemokinesis", "Brand", "Feed", "Pact's End",
        ],
        "B": [
            "Uppercut", "Headbutt", "Iron Wave", "Body Slam",
            "Breakthrough", "Armaments", "Bludgeon",
            "Bloodletting", "Inferno", "Juggernaut",
        ],
        "avoid": [
            "Anger", "Setup Strike", "Clash",
        ],
    },
    "silent": {
        # Updated April 2026 from wiki.gg, Mobalytics, Nat1Gaming, PCGamesN guides.
        # Sly is the strongest keyword — Master Planner + discard enablers dominate.
        # Adrenaline is universally S-tier. After Image is elite with Shivs.
        # Poison scales exponentially (n*(n+1)/2 total damage) — Noxious Fumes is
        # the best sustained damage card. Keep decks THIN for Sly cycling.
        "S": [
            "Adrenaline",                 # Draw 2 + 1 energy, zero cost — universally #1
            "Master Planner",             # Gives all Skills Sly — linchpin of best archetype
            "Well-Laid Plans",            # Retain best card — critical for consistency
            "Tools of the Trade",         # Draw + discard every turn — Sly engine
            "Footwork",                   # Dexterity scaling — core block card
            "Noxious Fumes",              # Auto-poison every turn — best sustained damage
            "Wraith Form",                # Intangible — the best defensive card in the game
        ],
        "A": [
            "Backflip",                   # Draw 2 + block — best utility card
            "Acrobatics",                 # Draw 3 discard 1 — top draw + Sly trigger
            "Dash",                       # 10 damage + 10 block — best Act 1 card
            "Leg Sweep",                  # 12 damage + 12 block + Weak — top defensive
            "Backstab",                   # 11 free damage turn 1 — huge Act 1 tempo
            "Blade Dance",                # 3-4 Shivs — defines the Shiv engine
            "Cloak and Dagger",           # Shivs + block — dual purpose
            "Calculated Gamble",          # Mass discard + redraw — Sly combo enabler
            "Tactician",                  # Energy on discard — Sly economy card
            "Accuracy",                   # Shivs deal +4 — but only in Shiv decks
            "Infinite Blades",            # Free Shiv per turn — Shiv engine starter
            "Deadly Poison",              # 5 base poison — efficient poison application
            "Bouncing Flask",             # Poison to random enemies — good AoE poison
            "Burst",                      # Double next skill — combo multiplier
            "Dodge and Roll",             # Block now + next turn — good defense
            "Prepared",                   # Draw 2 discard 1 — cheap Sly enabler
        ],
        "B": [
            "Knife Trap",                 # Conditional Shiv generation
            "Dagger Throw",               # Draw + discard — cycling utility
            "Reflex",                     # Draw on discard — Sly synergy piece
            "Poisoned Stab",              # Damage + poison — decent filler
            "Finisher",                   # Scales with cards played — Shiv payoff
            "Hidden Daggers",             # Shiv generation
            "Fan of Knives",              # AoE damage + draw — multi-enemy utility
            "Deflect",                    # Free block — never bad
            "Ricochet", "Speedster", "Abrasive", "Haze",
            "Outbreak", "Bubble Bubble", "Mirage",
            "Untouchable", "Flick-Flack",
            "Accelerant",                 # Poison stays up — good in poison decks
            "Storm of Steel",             # Discard hand for Shivs — Sly combo finisher
            "Serpent Form",               # Weak scaling — long fights only
            "Snakebite",                  # 7 poison — solid single-target poison
            "Corrosive Wave",             # Draw-triggered AoE poison — strong in draw decks
        ],
        "avoid": [
            "Slice",                      # Worse Strike
        ],
    },
}


def format_tier_list(character: str = "ironclad") -> str:
    """Format the tier list as a compact string for prompts."""
    tiers = CARD_TIERS.get(character, CARD_TIERS["ironclad"])
    lines = []
    for tier, cards in tiers.items():
        if tier == "avoid":
            lines.append(f"AVOID: {', '.join(cards)}")
        else:
            lines.append(f"{tier}-tier: {', '.join(cards)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Relic guides — per character
# Helps advisor evaluate relic picks (boss relics, events, shops).
# ---------------------------------------------------------------------------

RELIC_GUIDE: dict[str, dict[str, dict]] = {
    "ironclad": {
        # Based on Mobalytics Ironclad guide + gameplay analysis.
        "top_picks": {
            "note": "Universally strong — take in almost any deck",
            "relics": [
                "Charon's Ashes",     # AoE damage on exhaust — insane with Corruption
                "Tungsten Rod",       # Reduces ALL HP loss by 1 — stacks with everything
                "Paper Krane",        # Weak reduces damage to 60% instead of 75%
                "Ice Cream",          # Unspent energy carries over — enables big turns
                "Mummified Hand",     # Free energy on power play — snowball engine
                "Chemical X",         # +2 to all X-cost cards (Whirlwind!)
                "Demon Tongue",       # Heal on HP-spend cards — amazing with Offering/Bloodletting
                "Tough Bandages",     # Block on discard — great with exhaust & Burning Pact
            ],
        },
        "strength_scaling": {
            "note": "Strong in Strength decks (Demon Form, Inflame, Spot Weakness)",
            "relics": [
                "Brimstone",          # +2 Str to you AND enemies — risk/reward, favors Str decks
                "Ruined Helmet",      # Usually +2 Str — not build-defining but solid
                "Sword of Jade",      # Free 3 Str — always good
                "Vajra",              # +1 Str at combat start
                "Anchor",             # 10 Block turn 1 — buys time to set up Demon Form
                "Horn Cleat",         # 14 Block turn 1 — same idea, even better
                "Permafrost",         # Retain 1 card turn 1 — helps keep key setup cards
            ],
        },
        "block_build": {
            "note": "Strong in Block decks (Barricade, Juggernaut, Body Slam)",
            "relics": [
                "Cloak Clasp",        # Block on empty hand — triggers Juggernaut
                "Fresnel Lens",       # Boosts Block gained from cards
                "Vambrace",           # Works like Unmovable — persistent Block
                "Sai",                # Simple Block generation on attacks
                "Parrying Shield",    # Extra damage from Block surplus
                "Pael's Legion",      # Block that adds up, especially with Barricade
                "Bronze Scales",      # Thorns — good if you can tank hits
                "Self-Forming Clay",  # Block when losing HP — decent safety net
            ],
        },
        "exhaust_engine": {
            "note": "Strong in Exhaust decks (Corruption, Feel No Pain, Dark Embrace)",
            "relics": [
                "Charon's Ashes",     # AoE damage per exhaust — top-tier
                "Forgotten Soul",     # Smaller-scale exhaust synergy
                "Burning Sticks",     # Smaller-scale Dead Branch effect
                "Joss Paper",         # Extra draw on exhaust
                "Tough Bandages",     # Block on discard/exhaust
            ],
        },
        "hp_spend": {
            "note": "Strong with HP-spending cards (Offering, Bloodletting, Hemokinesis)",
            "relics": [
                "Demon Tongue",       # Heal when spending HP — top-tier here
                "Centennial Puzzle",  # Draw on HP loss — often triggers turn 1
                "Self-Forming Clay",  # Block when losing HP
                "Red Skull",          # +3 Str when below 50% HP
            ],
        },
        "avoid": {
            "note": "Relics with downsides that usually aren't worth it",
            "relics": [
                "Philosopher's Stone", # +1 Str to ALL enemies — too dangerous
                "Ectoplasm",          # Can't gain gold — cripples shop pathing
                "Velvet Choker",      # 6-card play limit — ruins exhaust/Corruption
                "Sozu",               # Can't gain potions — potions save runs
            ],
        },
    },
    "silent": {
        # Based on Mobalytics Silent guide.
        "top_picks": {
            "note": "Universally strong — take in almost any Silent deck",
            "relics": [
                "Paper Krane",        # Weak reduces damage to 60% — Silent applies Weak easily
                "Ice Cream",          # Unspent energy carries over — enables big Shiv/combo turns
                "Mummified Hand",     # Free energy on power play — Accuracy/Infinite Blades
                "Tungsten Rod",       # Reduces ALL HP loss by 1
            ],
        },
        "shiv_synergy": {
            "note": "Strong in Shiv decks (Accuracy, Infinite Blades, Blade Dance)",
            "relics": [
                "Shuriken",           # Gain Strength from playing Attacks — Shivs trigger this
                "Kunai",              # Gain Dexterity from playing Attacks — Shivs trigger this
                "Ornamental Fan",     # Gain Block from playing Attacks — Shivs trigger this
                "Nunchaku",           # Gain energy from playing Attacks
                "Ninja Scroll",       # Start combat with 3 Shivs
                "Kusarigama",         # Works with Shiv spam
                "Joss Paper",         # Extra draw on exhaust — Shivs exhaust
            ],
        },
        "poison_synergy": {
            "note": "Strong in Poison decks (Noxious Fumes, Deadly Poison, Catalyst)",
            "relics": [
                "Snecko Skull",       # Extra Poison on application
                "Twisted Funnel",     # Apply Poison at combat start
                "Unsettling Lamp",    # Doubles first Poison hit
                "Anchor",             # 10 Block turn 1 — survive while Poison ramps
                "Horn Cleat",         # 14 Block turn 1 — survive while Poison ramps
                "Captain's Wheel",    # Defensive coverage for slow starts
            ],
        },
        "sly_synergy": {
            "note": "Strong in Sly/discard decks (Tactician, Reflex, Calculated Gamble)",
            "relics": [
                "Tingsha",            # Damage on discard — great with high discard volume
                "Tough Bandages",     # Block on discard — strong cycling defense
                "The Abacus",         # Extra Block generation on shuffle
            ],
        },
        "avoid": {
            "note": "Relics with downsides that hurt Silent",
            "relics": [
                "Velvet Choker",      # 6-card play limit — ruins Shiv spam and Sly cycling
                "Philosopher's Stone", # +1 Str to ALL enemies — Silent has low HP
                "Ectoplasm",          # Can't gain gold — cripples shop pathing
                "Sozu",               # Can't gain potions — potions save runs
            ],
        },
    },
}


def format_relic_guide(character: str = "ironclad") -> str:
    """Format the relic guide as a compact string for prompts."""
    guide = RELIC_GUIDE.get(character, RELIC_GUIDE["ironclad"])
    lines = []
    for category, info in guide.items():
        label = category.replace("_", " ").upper()
        relic_names = [r.split("  ")[0] for r in info["relics"]]  # strip comments
        lines.append(f"{label} ({info['note']}): {', '.join(relic_names)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-character config — key cards, removal priorities, etc.
# ---------------------------------------------------------------------------

CHARACTER_CONFIG: dict[str, dict] = {
    "ironclad": {
        "key_card": "Bash",
        "key_card_reason": "it's your only source of Vulnerable (50% more damage)",
        "removal_priority": ["Strike", "Defend"],
    },
    "silent": {
        "key_card": "Survivor",
        "key_card_reason": "it enables Sly discards and provides Block",
        "protect_cards": ["Survivor", "Neutralize"],  # Never remove/transform these
        "removal_priority": ["Strike", "Defend"],     # Strikes first — they dilute draws
    },
}


# ---------------------------------------------------------------------------
# Strategy parameters — advisor behavior (character-agnostic)
# ---------------------------------------------------------------------------

STRATEGY = {
    # ── A/B midpoints (2026-04-11) ──
    # Midpoints of old Champion (A) and Challenger (B) tunings.
    # Identical to config_b.py STRATEGY — the experiment is routing-only.
    # NOTE: In profile A (Self-Play), most of these thresholds are
    # vestigial for map/rest/shop because the network handles those
    # screens directly. They still matter for deterministic fallbacks
    # when the network returns None.

    # Deck size thresholds
    "deck_lean_target": 11,          # midpoint of 12 / 10
    "deck_warn_threshold": 14,       # midpoint of 15 / 13

    # HP thresholds for map decisions
    "hp_critical_pct": 0.375,        # midpoint of 0.35 / 0.40
    "hp_low_pct": 0.575,             # midpoint of 0.55 / 0.60
    "hp_elite_min_pct": 0.775,       # midpoint of 0.75 / 0.80

    # Rest site thresholds
    "rest_heal_threshold": 0.45,     # midpoint of 0.40 / 0.50
    "rest_upgrade_threshold": 0.75,  # midpoint of 0.70 / 0.80
    "boss_rest_threshold": 0.75,     # midpoint of 0.70 / 0.80

    # Shop behavior
    "auto_remove_at_shop": True,     # unchanged
    "shop_max_advisor_calls": 3,     # unchanged

    # Boss floors (for pre-boss logic)
    "boss_floors": {15, 16, 33, 34, 51, 52},
}


def detect_character(state: dict) -> str:
    """Extract character key from game state. Defaults to 'ironclad'."""
    run = state.get("run") or {}
    name = (run.get("character_name") or run.get("character_id") or "").lower()
    if "silent" in name:
        return "silent"
    if "ironclad" in name:
        return "ironclad"
    return "ironclad"
