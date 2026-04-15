"""Config Profile B — "Deterministic Baseline" (rule-based routing for non-combat).

Experiment design (2026-04-11):
  Profile B is the deterministic baseline in the 2026-04-11 A/B experiment.
  It sets USE_NETWORK_ROUTING = False, so every non-combat screen is
  driven by the rule-based deterministic_advisor with no network attempt.
  Combat still uses MCTS (the network's combat head is not part of the
  experiment — we know it beats any deterministic combat alternative).

  Profile A (config_a.py) is the "Self-Play" profile that sets
  USE_NETWORK_ROUTING = True and leans on the network's option head
  first for map/rest/shop/Neow, falling back to deterministic only when
  the network returns None.

  Both profiles carry IDENTICAL scalar weights — EVALUATOR and STRATEGY
  values are midpoints between the old Champion and Challenger tunings.
  The ONLY experimental variable is routing, so any win-rate difference
  should cleanly attribute to "network option head" vs "deterministic
  advisor" for the three screens where routing differs.

  Universal network routes (both profiles, NOT gated by USE_NETWORK_ROUTING):
    - Card reward: network option head picks take-card-N / skip.
      Falls back to decide_card_reward (organic picker) on network failure.
    - Act 1+ events: network option head picks among unlocked event
      options. Falls back to decide_event_default (sim scorer) on
      network failure.

  Residual value differences:
    - CARD_TIERS lists still differ between profiles. Profile B keeps
      the "Survive & Scale" tier list (kept here) to preserve its
      elite/boss survival lessons as a fallback signal if the network
      card-reward handler ever bails. Profile A keeps the Champion
      tier list. In normal operation both profiles defer to the
      network card-reward handler, so the tier list is a fallback.

Run: bash play.sh --profile b   (or set STS2_CONFIG_PROFILE=b)
Compare: bash play.sh --profile a

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

See IMPROVEMENTS.md at the repo root for the full gap list and fixes.
===========================================================================
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Routing flag — the core A/B experimental variable.
# ---------------------------------------------------------------------------
# False → live play skips the network entirely for non-combat screens and
#         goes straight to the deterministic advisor. This is the
#         "Deterministic Baseline" profile. Combat is still MCTS.
# Profile A (config_a.py) sets this to True (Self-Play).
# Consumed by runner.py's non-combat screen dispatcher.
USE_NETWORK_ROUTING: bool = False


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
    # Challenger "Survive & Scale" (B) tunings. config_a.py and
    # config_b.py carry IDENTICAL values — the A/B experiment is now
    # about ROUTING (USE_NETWORK_ROUTING flag), not weights.

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
        # ── Defensive scaling (these keep you alive) ──
        "Footwork": 10.0,           # ⬆ was 8.0 — THE most important Silent power;
                                    #   each stack adds block to every block card
        "Noxious Fumes": 10.0,      # ⬆ was 8.0 — AoE poison every turn; the longer
                                    #   the fight, the more value; crushes bosses
        "Well-Laid Plans": 8.0,     # ⬆ was 7.0 — retain your best card each turn;
                                    #   critical for playing the right card at the right time
        "Afterimage": 7.0,          # ⬆ was 5.0 — block on every card played; adds up
                                    #   fast in a high-card-play Silent deck

        # ── Offensive scaling (these win fights) ──
        "Accuracy": 8.0,            # ⬇ was 10.0 — Shivs are strong but slow vs bosses;
                                    #   defensive scaling matters more for survival
        "Infinite Blades": 7.0,     # ⬇ was 9.0 — 1 Shiv/turn is good but not game-winning
        "Accelerant": 8.0,          # ⬆ was 7.0 — poison doubler; key for boss kills
        "Serpent Form": 7.0,        # (unchanged) — strong damage output

        # ── Engine powers (these make your deck work) ──
        "Tools of the Trade": 9.0,  # ⬆ was 8.0 — draw + discard each turn is the best
                                    #   Silent engine; finds key cards, enables Sly
        "Master Planner": 6.0,      # (unchanged)
        "Abrasive": 5.0,            # (unchanged)
        "Envenom": 4.0,             # (unchanged)
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
        # CHALLENGER: "Survive & Scale" tier list.
        #
        # Philosophy: Silent wins by NOT dying. Boss fights are long (8+ turns),
        # and the encounter report shows 0% boss win rate. We need cards that
        # (a) generate block consistently and (b) scale damage over time.
        #
        # Key changes vs champion:
        #   - Defensive all-stars promoted to S: Leg Sweep, Dodge and Roll
        #   - Catalyst promoted to A: makes Poison viable as boss-killer
        #   - Backflip added to A: draw + block = Silent's ideal card
        #   - Accuracy/Infinite Blades demoted to A: Shivs are slow vs bosses
        #   - More cards in avoid: keep the deck lean (target: 10 cards)
        "S": [
            "Footwork",                   # +Dexterity → every block card gets better forever
            "Leg Sweep",                  # ⬆ from A — 12 dmg + 12 block + Weak in one card;
                                          #   solves offense AND defense simultaneously
            "Dash",                       # 10 damage + 10 block — best Act 1 card
            "Well-Laid Plans",            # Retain best card — play the right answer each turn
            "Noxious Fumes",              # ⬆ from A — AoE poison scales every turn;
                                          #   the only way to kill bosses without burst damage
            "Tools of the Trade",         # Draw + discard engine — finds your key cards
            "Dodge and Roll",             # ⬆ from A — block this turn AND next turn;
                                          #   exactly what you need in long boss fights
            "Wraith Form",                # Intangible — take 1 dmg per hit for 2-3 turns;
                                          #   the best defensive card in the game
            "Adrenaline",                 # 0 cost, +2 energy, draw 2 — tempo king
        ],
        "A": [
            "Accuracy",                   # ⬇ from S — Shivs need setup and are slow vs bosses
            "Infinite Blades",            # ⬇ from S — 1 Shiv/turn is okay, not game-winning
            "Master Planner",             # ⬇ from S — good but not essential early
            "Knife Trap",                 # ⬇ from S — situational; strong but not core
            "Backstab",                   # 11 free damage turn 1 — huge early game tempo
            "Accelerant",                 # Poison triggered extra times — key for boss kills
            "Deadly Poison",              # Core poison card — 5 stacks = 15 future damage
            "Cloak and Dagger",           # Block + Shivs — does both jobs
            "Blade Dance",                # Shiv generation — good with Accuracy
            "Acrobatics",                 # Draw 3, discard 1 — great card flow
            "Backflip",                   # ⬆ NEW — draw 2 + gain block; exactly what
                                          #   Silent wants: defense that doesn't cost tempo
            "Untouchable",                # Strong defensive option
            "Flick-Flack",               # Multi-hit + block — versatile
            "Burst",                      # Double next skill — incredible with Catalyst/Leg Sweep
            "Serpent Form",               # Strong sustained damage
            "Deflect",                    # Free block — zero-cost = always playable
            "Calculated Gamble",          # Full hand refresh — powerful with lean deck
            "Tactician",                  # Energy on discard — fuels big turns
            "Malaise",                    # X-cost Str reduction + Weak — shuts down bosses
            "Bullet Time",               # All cards cost 0 this turn — explosive combos
            "Piercing Wail",             # ALL enemies -6 Str — huge defensive swing
            "Blur",                       # Block carries over — compounds with Footwork
            "Escape Plan",               # 0 cost draw + block — free cycle + defense
            "Predator",                   # 15 dmg + draw 2 — offense + card flow
            "Bouncing Flask",            # 9 poison spread — strong AoE poison
            "Pounce",                     # 12 dmg + next skill costs 0 — great tempo
            "Snakebite",                 # 7 poison for 2 energy — efficient poison source
            "Blade of Ink",              # +Str on attack play — scaling for Shiv builds
            "Corrosive Wave",            # Poison on draw — passive scaling engine
            "Echoing Slash",             # 10 AoE dmg per enemy — strong multi-enemy clear
            "Tracking",                   # Weak enemies take double damage — synergy payoff
        ],
        "B": [
            "Leading Strike",             # ⬇ from A — damage only, doesn't block
            "Poisoned Stab",              # ⬇ from A — too little poison to matter
            "Dagger Throw", "Ricochet", "Prepared", "Reflex",
            "Speedster", "Abrasive", "Haze", "Outbreak",
            "Bubble Bubble", "Mirage", "Fan of Knives",
            "Hidden Daggers", "Finisher", "Afterimage",
            "Dagger Spray",              # 4 AoE dmg × 2 — decent hallway clear
            "Anticipate",                 # Temp 3 Dex — good for one big block turn
            "Precise Cut",               # 15 dmg conditional — unreliable
            "Memento Mori",              # 12 dmg + discard bonus — deck-dependent
            "Strangle",                   # 8 dmg + damage on card play — slow setup
            "Hand Trick",                # 7 block + Sly — decent utility
            "Flechettes",               # Dmg per skill in hand — variable value
            "Follow Through",            # 6 AoE conditional — needs setup
            "Skewer",                     # X-cost multi-hit — energy hungry
            "Pinpoint",                   # 17 dmg w/ cost reduction — situational
            "Expertise",                  # Draw to 6 — good in small hands only
            "Up My Sleeve",              # 3 Shivs — okay with Accuracy, mediocre without
            "Phantom Blades",            # Shiv retain — niche Shiv synergy
            "Expose",                     # Strip artifact + block — utility vs elites
            "Shadowmeld",                # Double block — good with Footwork
            "Storm of Steel",            # Discard hand → Shivs — risky, high ceiling
            "The Hunt",                   # 10 dmg + conditional reward — inconsistent
            "Murder",                     # 2+ scaling dmg — slow buildup
            "Shadow Step",               # Double attack next turn — setup required
        ],
        "avoid": [
            "Slice",                      # 6 damage for 0 cost — but adds junk to deck
            "Sucker Punch",               # 7 dmg + 1 Weak is not enough impact
            "Flanking",                   # Multiplayer only — useless in solo
            "Sneaky",                     # Multiplayer only — useless in solo
            "Grand Finale",              # Must empty draw pile — far too conditional
            "Nightmare",                  # Cost 3, needs specific card in hand — too slow
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
    # All scalars below are midpoints between the old Champion (A) and
    # Challenger "Survive & Scale" (B) tunings. config_a.py and
    # config_b.py carry IDENTICAL STRATEGY values — the A/B experiment
    # is about ROUTING (USE_NETWORK_ROUTING), not weights.

    # ── Deck size ──
    "deck_lean_target": 11,          # midpoint of 12 / 10
    "deck_warn_threshold": 14,       # midpoint of 15 / 13

    # ── HP thresholds for map decisions ──
    "hp_critical_pct": 0.375,        # midpoint of 0.35 / 0.40
    "hp_low_pct": 0.575,             # midpoint of 0.55 / 0.60
    "hp_elite_min_pct": 0.775,       # midpoint of 0.75 / 0.80

    # ── Rest site thresholds ──
    "rest_heal_threshold": 0.45,     # midpoint of 0.40 / 0.50
    "rest_upgrade_threshold": 0.75,  # midpoint of 0.70 / 0.80
    "boss_rest_threshold": 0.75,     # midpoint of 0.70 / 0.80

    # ── Shop behavior ──
    "auto_remove_at_shop": True,     # (unchanged) — removing Strikes is always good
    "shop_max_advisor_calls": 3,     # (unchanged)

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
