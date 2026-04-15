"""Relic effect registry.

Central place for how every relic is simulated by the combat engine and
by the run setup in ``alphazero/full_run.py``.  Most relics are
data-only entries in the tables below; genuinely one-off behaviour
lives in small helper functions at the bottom of the file.

Design notes
------------
* Combat hooks are:
    ``apply_start_of_combat(state, is_elite=False, is_boss=False)``
    ``apply_turn_start(state)``
    ``apply_card_play(state, card)``
    ``apply_end_turn(state)``
    ``apply_end_combat(state)``
    ``get_damage_multiplier(relics)``
    ``get_block_multiplier(relics)``
    ``get_incoming_damage_reduction(relics)``

* Every simulated relic has a data-only entry in at least one table, so
  "what does this relic do" is answerable by reading this module front
  to back.

* For relics whose real effect is impossible or expensive to simulate
  (deck upgrades, card transforms, event-only rewards), we use a
  "proxy" signal: a small damage or block multiplier applied globally
  while the relic is owned.  The goal is for the agent to learn that
  the relic has value, without lying to combat_engine about state it
  doesn't track.

* Numbers in the tables are intentionally modest.  The RL loop will
  find the right weights; the job of this module is to make sure no
  relic is silently a no-op.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .models import Card, CombatState


# =============================================================================
# START-OF-COMBAT PASSIVES
# =============================================================================
#
# Fires exactly once at the beginning of combat.  Keys in the inner
# dicts map to standard effect primitives:
#     block, thorns, strength, dexterity, vigor, heal, energy, draw,
#     damage_all, vulnerable_all, weak_all, poison_all, plating,
#     shivs_to_hand, self_damage, max_hp_this_fight
#
# Elite / boss conditional passives live in ELITE_ONLY_START and
# BOSS_ONLY_START so apply_start_of_combat can dispatch them separately.

START_OF_COMBAT: dict[str, dict[str, int]] = {
    # --- existing 17 (direct ports) ---
    "ANCHOR":            {"block": 10},
    "BLOOD_VIAL":        {"heal": 2},
    "BAG_OF_PREPARATION":{"draw": 2},
    "BRONZE_SCALES":     {"thorns": 3},
    "BAG_OF_MARBLES":    {"vulnerable_all": 1},
    "FESTIVE_POPPER":    {"damage_all": 9},
    "LANTERN":           {"energy": 1},
    "ODDLY_SMOOTH_STONE":{"dexterity": 1},
    "RING_OF_THE_SNAKE": {"draw": 2},
    # --- new commons ---
    "VAJRA":             {"strength": 1},
    "GORGET":            {"block": 4},  # "Plating" → modelled as Block
    # --- uncommons ---
    "AKABEKO":           {"vigor": 8},  # treated as first-attack damage bonus (vigor applied via Strength proxy below)
    "RED_MASK":          {"weak_all": 1},
    "TWISTED_FUNNEL":    {"poison_all": 4},
    "PETRIFIED_TOAD":    {},  # handled via GLOBAL_DAMAGE_MULTIPLIERS proxy
    # --- rares ---
    "VEXING_PUZZLEBOX":  {},  # free random card; handled via GLOBAL_DAMAGE_MULTIPLIERS
    # --- shop ---
    "NINJA_SCROLL":      {"shivs_to_hand": 3},
    "RINGING_TRIANGLE":  {},  # retain hand turn 1; multiplier proxy
    # --- event ---
    "FAKE_ANCHOR":       {"block": 4},
    "FAKE_BLOOD_VIAL":   {"heal": 1},
    "EMBER_TEA":         {"strength": 2},          # first 2 combats; modelled always-on (proxy)
    "SWORD_OF_JADE":     {"strength": 3},
    "BONE_TEA":          {},  # upgrade starting hand; damage mult proxy
    "ROYAL_POISON":      {"self_damage": 4},
    # --- ancient (boss) ---
    "DELICATE_FROND":    {},  # fill potion slots; handled via OOC potions
    "CROSSBOW":          {},  # per-turn effect; lives in TURN_START
    "JEWELED_MASK":      {},  # free power in hand; damage mult proxy
    "CHOICES_PARADOX":   {},  # random card to hand; damage mult proxy
    "RADIANT_PEARL":     {},  # luminesce to hand; damage mult proxy
    "NEOWS_TORMENT":     {},  # Neow's Fury in deck; damage mult proxy
    "STORYBOOK":         {},
    # --- cosmicImpetus/rune-based (summon/channel proxies) ---
    "DATA_DISK":         {},  # Channel 1 Focus; damage mult proxy
    "RUNIC_CAPACITOR":   {},  # Channel 3 Orb slots; damage mult proxy
}

# Akabeko's "next attack deals 8 extra" isn't a Power in our engine;
# treat it as +1 Strength for this fight so the signal flows through
# calculate_attack_damage without special handling.
AKABEKO_STRENGTH_PROXY = 1

# Additional start-of-combat passives that only fire against elites.
ELITE_ONLY_START: dict[str, dict[str, int]] = {
    "SLING_OF_COURAGE": {"strength": 2},
    "BOOMING_CONCH":    {"draw": 2},
}

# Additional start-of-combat passives that only fire against bosses.
BOSS_ONLY_START: dict[str, dict[str, int]] = {
    "PANTOGRAPH":  {"heal": 25},
    "STONE_CRACKER": {},  # proxy via damage mult
}


# =============================================================================
# TURN-START EFFECTS
# =============================================================================
#
# List of effects that fire at the beginning of every player turn.
# An entry may include a ``turn`` key to limit to a specific turn number,
# or ``turn_min`` / ``turn_max`` for ranges.
#
# Energy gains are used for relics whose "gain 1 energy at start of
# turn" effect is the whole relic.  For drawback-carrying ones (Sozu,
# Ectoplasm, Spiked Gauntlets, Blessed Antler, Velvet Choker, Blood
# Soaked Rose, Pumpkin Candle, Philosopher's Stone, Whispering Earring,
# Paels Flesh) we grant the bonus as a proxy and ignore the drawback —
# this is the "hard relics get a positive signal even if we can't
# simulate them" tier.

TURN_START: dict[str, list[dict]] = {
    # Existing 17
    "ART_OF_WAR":        [{"_marker": "art_of_war"}],  # handled specially

    # Easy flat bonuses
    "MERCURY_HOURGLASS": [{"damage_all": 3}],
    "SAI":               [{"block": 7}],
    "PAELS_BLOOD":       [{"draw": 1}],
    "FIDDLE":            [{"draw": 2}],
    "SNECKO_EYE":        [{"draw": 2}],  # drawback (Confused) ignored — proxy tier

    # Turn-conditional bonuses
    "CANDELABRA":        [{"turn": 2, "energy": 1}],
    "HORN_CLEAT":        [{"turn": 2, "block": 14}],
    "SPARKLING_ROUGE":   [{"turn": 3, "strength": 1, "dexterity": 1}],
    "CAPTAINS_WHEEL":    [{"turn": 3, "block": 18}],
    "CHANDELIER":        [{"turn": 3, "energy": 1}],
    "STONE_CALENDAR":    [{"turn": 7, "damage_all": 52}],
    "PAELS_FLESH":       [{"turn_min": 3, "energy": 1}],
    "BREAD":             [{"turn": 1, "energy": -1},
                          {"turn_min": 2, "energy": 1}],

    # Draft-me-an-energy-per-turn relics (proxy: ignore their drawback)
    "ECTOPLASM":         [{"energy": 1}],
    "SOZU":              [{"energy": 1}],
    "SPIKED_GAUNTLETS":  [{"energy": 1}],
    "BLESSED_ANTLER":    [{"energy": 1}],
    "BLOOD_SOAKED_ROSE": [{"energy": 1}],
    "PUMPKIN_CANDLE":    [{"energy": 1}],
    "PHILOSOPHERS_STONE":[{"energy": 1}],
    "WHISPERING_EARRING":[{"energy": 1}],
    "VELVET_CHOKER":     [{"energy": 1}],  # 6-cards-per-turn drawback ignored
    "DIVINE_DESTINY":    [{"energy": 1}],
    "VERY_HOT_COCOA":    [{"turn": 1, "energy": 1}],  # "additional energy" on turn 1 only
    "INFUSED_CORE":      [{"turn": 1, "block": 9}],   # "Channel 3 Lightning" ≈ 9 block proxy
    "PHYLACTERY_UNBOUND":[{"turn": 1, "draw": 2}, {"draw": 1}],  # summon proxy → more draws
    "RING_OF_THE_DRAKE": [{"turn_max": 3, "draw": 2}],

    # Damage-each-turn relics
    "MR_STRUGGLES":      [{"_marker": "mr_struggles"}],
    "CROSSBOW":          [{"_marker": "crossbow"}],    # free random attack
    "TOASTY_MITTENS":    [{"strength": 1}],             # exhaust top proxy → net +str only
    "HISTORY_COURSE":    [],  # proxy via GLOBAL_DAMAGE_MULTIPLIERS
    "POLLINOUS_CORE":    [{"_marker": "pollinous"}],
    "SEAL_OF_GOLD":      [{"energy": 1}],  # spend-gold-for-energy proxy

    # --- missing rune-based relics (summon/strength/star proxies) ---
    "BOUND_PHYLACTERY":  [{"draw": 1}],  # Summon 1; proxy via extra draw
    "BRIMSTONE":         [{"strength": 1}],            # gain Strength; direct effect
    "EMOTION_CHIP":      [{"draw": 1}],                # conditional passive; proxy via draw

    "BLACK_BLOOD":       [],                            # end-of-combat only
}


# =============================================================================
# CARD-PLAY TRIGGERS
# =============================================================================
#
# Counter-based triggers that accumulate across plays.  ``match`` picks
# which cards count, ``every`` is the threshold, ``scope`` is where the
# counter resets, and ``effect`` is what fires when the counter hits.
#
# match values:
#   "attack"         — any Attack card
#   "skill"          — any Skill card
#   "power"          — any Power card
#   "any"            — any card
#   "upgraded_attack"— Upgraded Attacks (for Miniature Cannon)
#   "strike_in_name" — cards with "Strike" in the name (Strike Dummy)
#   "expensive"      — cards with effective cost >= 2 (Intimidating Helmet)
#   "shiv"           — Shiv cards (Helical Dart)
#   "discard"        — handled in the discard helper, not here
#
# scope values:
#   "turn"    — counter resets at start of turn
#   "combat"  — counter resets at start of combat
#
# effect keys:
#   strength, dexterity, strength_this_turn, dexterity_this_turn,
#   block, block_this_turn, energy, draw,
#   damage_all, damage_random

CARD_PLAY_TRIGGERS: dict[str, dict] = {
    # --- existing 17 ---
    "KUNAI":         {"match": "attack", "every": 3, "scope": "turn",
                      "effect": {"dexterity": 1}},
    "SHURIKEN":      {"match": "attack", "every": 3, "scope": "turn",
                      "effect": {"strength": 1}},
    "ORNAMENTAL_FAN":{"match": "attack", "every": 3, "scope": "turn",
                      "effect": {"block": 4}},
    "LETTER_OPENER": {"match": "skill",  "every": 3, "scope": "turn",
                      "effect": {"damage_all": 5}},
    "NUNCHAKU":      {"match": "attack", "every": 10,"scope": "combat",
                      "effect": {"energy": 1}},

    # --- new per-card-type triggers ---
    "KUSARIGAMA":    {"match": "attack", "every": 3, "scope": "turn",
                      "effect": {"damage_random": 6}},
    "TUNING_FORK":   {"match": "skill",  "every": 10,"scope": "combat",
                      "effect": {"block": 7}},
    "IRON_CLUB":     {"match": "any",    "every": 4, "scope": "combat",
                      "effect": {"draw": 1}},
    "PEN_NIB":       {"match": "attack", "every": 10,"scope": "combat",
                      "effect": {"damage_all": 10}},  # approx double-damage as flat AoE
    "JOSS_PAPER":    {"match": "exhaust","every": 5, "scope": "combat",
                      "effect": {"draw": 1}},
    "HELICAL_DART":  {"match": "shiv",   "every": 1, "scope": "always",
                      "effect": {"dexterity_this_turn": 1}},
    "GAME_PIECE":    {"match": "power",  "every": 1, "scope": "always",
                      "effect": {"draw": 1}},
    "PERMAFROST":    {"match": "power",  "every": 1, "scope": "combat",
                      "effect": {"block": 6}},
    "LOST_WISP":     {"match": "power",  "every": 1, "scope": "always",
                      "effect": {"damage_all": 8}},
    "MUMMIFIED_HAND":{"match": "power",  "every": 1, "scope": "always",
                      "effect": {"_marker": "mummified_hand"}},
    "DAUGHTER_OF_THE_WIND":{"match": "attack", "every": 1, "scope": "always",
                            "effect": {"block": 1}},

    # Damage / block bonuses on specific card shapes
    "STRIKE_DUMMY":    {"match": "strike_in_name",
                        "effect": {"damage_bonus": 3}},
    "MINIATURE_CANNON":{"match": "upgraded_attack",
                        "effect": {"damage_bonus": 3}},
    "INTIMIDATING_HELMET":{"match": "expensive",
                           "effect": {"block": 4}},

    # Trifecta / rainbow
    "RAINBOW_RING":  {"match": "trifecta", "every": 1, "scope": "turn",
                      "effect": {"strength": 1, "dexterity": 1}},

    # --- missing "whenever" triggers ---
    "BONE_FLUTE":    {"match": "power",  "every": 1, "scope": "always",
                      "effect": {"block": 2}},  # Osty attack proxy
    "CHARONS_ASHES": {"match": "exhaust", "every": 1, "scope": "always",
                      "effect": {"damage_random": 3}},
    "IVORY_TILE":    {"match": "expensive", "every": 1, "scope": "always",
                      "effect": {"energy": 1}},  # cards cost 3+ → energy bonus
    "REGALITE":      {"match": "power",  "every": 1, "scope": "always",
                      "effect": {"block": 2}},  # colorless creation proxy
    "SELF_FORMING_CLAY":{"match": "vulnerable_all", "every": 1, "scope": "combat",
                         "effect": {"block": 3}},  # HP loss proxy via vulnerable check
}


# =============================================================================
# DISCARD TRIGGERS
# =============================================================================
#
# Fired by effects that discard cards mid-turn (e.g. Calculated Gamble).
# Keys map to per-discard effect.

DISCARD_TRIGGERS: dict[str, dict] = {
    "TINGSHA":       {"damage_random": 3},
    "TOUGH_BANDAGES":{"block": 3},
}


# =============================================================================
# SHUFFLE TRIGGERS (fire whenever draw pile reshuffles)
# =============================================================================

SHUFFLE_TRIGGERS: dict[str, dict] = {
    "PENDULUM":  {"draw": 1},
    "THE_ABACUS":{"block": 6},
}


# =============================================================================
# EXHAUST TRIGGERS
# =============================================================================

EXHAUST_TRIGGERS: dict[str, dict] = {
    "FORGOTTEN_SOUL":{"damage_random": 1},
}


# =============================================================================
# ENEMY DEATH TRIGGERS
# =============================================================================

ENEMY_DEATH_TRIGGERS: dict[str, dict] = {
    "GREMLIN_HORN":{"draw": 1, "energy": 1},
}


# =============================================================================
# FIRST-EVENT-IN-COMBAT TRIGGERS
# =============================================================================
#
# These need a per-combat flag tracking whether they've fired yet.

FIRST_HP_LOSS_TRIGGERS: dict[str, dict] = {
    "CENTENNIAL_PUZZLE":{"draw": 3},
    "DEMON_TONGUE":     {"heal": 3},  # heal equal to damage taken; proxy as flat heal
    "RUINED_HELMET":    {"strength": 2},  # double strength gain; proxy via strength
}

FIRST_BLOCK_GAIN_DOUBLE: set[str] = {"VAMBRACE"}  # Already defined, but now actually used


# =============================================================================
# END-OF-TURN TRIGGERS
# =============================================================================
#
# These fire after card plays but before enemy intents resolve.

END_OF_TURN: dict[str, dict] = {
    # Existing
    "CLOAK_CLASP":     {"block_per_hand_card": 1},
    "ART_OF_WAR":      {"_marker": "art_of_war"},  # handled specially

    # New
    "ORICHALCUM":      {"if_no_block": {"block": 6}},
    "FAKE_ORICHALCUM": {"if_no_block": {"block": 3}},
    "RIPPLE_BASIN":    {"if_no_attacks": {"block": 4}},
    "PARRYING_SHIELD": {"if_block_at_least": (10, {"damage_random": 6})},
    "SCREAMING_FLAGON":{"if_empty_hand": {"damage_all": 20}},
    "POCKETWATCH":     {"if_played_at_most": (3, {"_marker": "pocketwatch_flag"})},
    "LUNAR_PASTRY":    {},  # gain Star at end-of-turn; proxy via GLOBAL_DAMAGE_MULTIPLIERS
}


# =============================================================================
# END-OF-COMBAT TRIGGERS
# =============================================================================

END_OF_COMBAT: dict[str, dict] = {
    "BURNING_BLOOD":    {"heal": 6},
    "BLACK_BLOOD":      {"heal": 12},
    "MEAT_ON_THE_BONE": {"heal_if_below_half": 12},
    "CHOSEN_CHEESE":    {"max_hp": 1},
}


# =============================================================================
# GLOBAL PROXY MULTIPLIERS
# =============================================================================
#
# Applied in effects.py's calculate_attack_damage / calculate_block_gain.
# These carry the weight of the "hard" and "weird" relic tiers —
# anything whose real effect we can't cheaply simulate (deck upgrades,
# card transforms, merchant discounts, reward enchants) gets a small
# positive signal so the agent still learns to value picking them up.
#
# Multipliers compound multiplicatively, but we cap them in the
# aggregator so a maximally stacked run can't run away.

GLOBAL_DAMAGE_MULTIPLIERS: dict[str, float] = {
    # Upgrade-on-pickup / upgrade-at-combat-start
    "BELLOWS":          1.04,
    "WHETSTONE":        1.03,
    "WAR_PAINT":        1.02,  # skill upgrade, mostly block
    "BONE_TEA":         1.03,
    "FRAGRANT_MUSHROOM":1.03,
    "STONE_CRACKER":    1.06,
    "GNARLED_HAMMER":   1.03,
    "PUNCH_DAGGER":     1.03,
    "TRI_BOOMERANG":    1.03,
    "NUTRITIOUS_SOUP":  1.03,
    "RAZOR_TOOTH":      1.04,  # per-combat upgrade of first attack/skill
    "SAND_CASTLE":      1.05,
    "YUMMY_COOKIE":     1.04,
    "ASTROLABE":        1.04,
    "POMANDER":         1.02,
    "MYSTIC_LIGHTER":   1.03,  # enchanted attacks
    "KIFUDA":           1.02,
    "ELECTRIC_SHRYMP":  1.02,
    "ROYAL_STAMP":      1.02,
    "GLITTER":          1.02,
    "WING_CHARM":       1.02,
    # Transform / reward improvers
    "PANDORAS_BOX":     1.06,
    "EMPTY_CAGE":       1.03,
    "PRECISE_SCISSORS": 1.01,
    "PRECARIOUS_SHEARS":1.02,
    "LEAFY_POULTICE":   1.02,
    "NEW_LEAF":         1.01,
    "ARCHAIC_TOOTH":    1.02,
    "TOUCH_OF_OROBAS":  1.02,
    # Free / bonus cards
    "VEXING_PUZZLEBOX": 1.02,
    "CHOICES_PARADOX":  1.02,
    "RADIANT_PEARL":    1.02,
    "NEOWS_TORMENT":    1.02,
    "STORYBOOK":        1.02,
    "TANXS_WHISTLE":    1.02,
    "JEWELED_MASK":     1.03,
    "TOOLBOX":          1.02,
    "DOLLYS_MIRROR":    1.02,
    "BING_BONG":        1.02,
    "BOOK_OF_FIVE_RINGS":1.01,
    "BYRDPIP":          1.02,
    # Rare / ancient reward cards
    "DUSTY_TOME":       1.04,
    "ARCANE_SCROLL":    1.04,
    "MASSIVE_SCROLL":   1.03,
    "LEAD_PAPERWEIGHT": 1.01,
    "SEA_GLASS":        1.03,
    "GLASS_EYE":        1.03,
    "ORRERY":           1.03,
    "LARGE_CAPSULE":    1.03,
    "SMALL_CAPSULE":    1.02,
    "DINGY_RUG":        1.01,
    # Elite / floor reward bonuses
    "WHITE_STAR":       1.02,
    "PRAYER_WHEEL":     1.02,
    "LAVA_LAMP":        1.02,
    "WAR_HAMMER":       1.03,
    "SILVER_CRUCIBLE":  1.03,
    "BLACK_STAR":       1.03,
    "LAVA_ROCK":        1.02,
    "LORDS_PARASOL":    1.04,
    "GOLDEN_COMPASS":   1.02,
    "DRIFTWOOD":        1.02,
    "WONGOS_MYSTERY_TICKET":1.02,
    # First-turn / first-card / first-play
    "RINGING_TRIANGLE": 1.02,
    "THROWING_AXE":     1.05,
    "MUSIC_BOX":        1.03,
    "BURNING_STICKS":   1.03,
    "UNSETTLING_LAMP":  1.03,
    # Per-combat card adds / enchants
    "BRILLIANT_SCARF":  1.03,
    "CHEMICAL_X":       1.03,
    "BIG_MUSHROOM":     1.02,
    # Stat padding proxies for cases where we can't fire the real effect
    "HAPPY_FLOWER":     1.02,
    "FAKE_HAPPY_FLOWER":1.01,
    "FAKE_STRIKE_DUMMY":1.01,
    "PETRIFIED_TOAD":   1.02,
    "HISTORY_COURSE":   1.05,
    "AKABEKO":          1.02,  # tiny bump beyond the Strength proxy
    # Unique ancient / meta
    "TOY_BOX":          1.03,
    "CLAWS":            1.02,
    "BEAUTIFUL_BRACELET":1.02,
    "FRESNEL_LENS":     1.02,
    "CURSED_PEARL":     1.01,
    "DISTINGUISHED_CAPE":1.01,
    "SERE_TALON":       1.01,
    "FUR_COAT":         1.03,
    "BIIIG_HUG":        1.01,
    "JEWELRY_BOX":      1.02,
    "PRISMATIC_GEM":    1.03,
    "PAELS_CLAW":       1.02,
    "PAELS_EYE":        1.02,
    "PAELS_GROWTH":     1.02,
    "PAELS_HORN":       1.01,
    "PAELS_WING":       1.02,
    "PAELS_TEARS":      1.02,
    "PAELS_TOOTH":      1.02,
    "PAELS_LEGION":     1.03,
    "NUTRITIOUS_OYSTER":1.01,
    "TEA_OF_DISCOURTESY":0.99,  # Daze drawback, tiny negative
    "DARKSTONE_PERIAPT":1.01,
    "MAW_BANK":         1.01,
    "DREAM_CATCHER":    1.01,
    "STONE_HUMIDIFIER": 1.01,
    "MEMBERSHIP_CARD":  1.02,
    "THE_COURIER":      1.02,
    "SHOVEL":           1.02,
    "GIRYA":            1.02,  # 3 free strength at rest sites
    "MINIATURE_TENT":   1.02,
    "DRAGON_FRUIT":     1.01,
    "DELICATE_FROND":   1.02,
    "FAKE_VENERABLE_TEA_SET":1.01,
    "VENERABLE_TEA_SET":1.02,
    "FAKE_SNECKO_EYE":  0.98,  # permanent Confused drawback
    "WHITE_BEAST_STATUE":1.02,
    "LUCKY_FYSH":       1.01,  # gold per card add
    "BOWLER_HAT":       1.01,  # 20% more gold
    "AMETHYST_AUBERGINE":1.01,
    "POTION_BELT":      1.02,
    "ALCHEMICAL_COFFER":1.02,
    "CAULDRON":         1.02,
    "LEES_WAFFLE":      1.01,
    "FAKE_LEES_WAFFLE": 1.005,
    "FAKE_MERCHANTS_RUG":1.00,  # literally does nothing — keep neutral-positive
    "WONGO_CUSTOMER_APPRECIATION_BADGE":1.00,
    "JUZU_BRACELET":    1.01,  # fewer ? rooms with fights
    "REGAL_PILLOW":     1.01,
    "TINY_MAILBOX":     1.01,
    "MEAL_TICKET":      1.01,
    "PLANISPHERE":      1.01,
    "ETERNAL_FEATHER":  1.01,
    "LASTING_CANDY":    1.02,
    "REPTILE_TRINKET":  1.02,  # +3 str per potion use
    "FROZEN_EGG":       1.04,
    "MOLTEN_EGG":       1.04,
    "TOXIC_EGG":        1.04,
    "GAMBLING_CHIP":    1.02,
    "ICE_CREAM":        1.04,
    "POCKETWATCH":      1.03,
    "UNCEASING_TOP":    1.02,
    "STURDY_CLAMP":     1.03,
    "LIZARD_TAIL":      1.02,  # life-save → proxy survival bonus
    "BEATING_REMNANT":  1.02,
    "PAPER_KRANE":      1.02,
    "OLD_COIN":         1.01,
    "SIGNET_RING":      1.01,
    "GOLDEN_PEARL":     1.01,
    "SNECKO_SKULL":     1.02,  # +1 poison on apply
    "CALLING_BELL":     1.03,
    "CURSED_KEY":       0.99,  # curse drawback if it exists
    "FIDDLE":           1.02,
    "BIG_HAT":          1.01,
    "BOOK_REPAIR_KNIFE":1.01,
    "BOOKMARK":         1.01,
    "BOOMING_CONCH":    1.01,
    "BOWLING_BALL":     1.01,
    "BREAST_PLATE":     1.01,
    "DIVINE_RIGHT":     1.02,
    "FENCING_MANUAL":   1.02,
    "GALACTIC_DUST":    1.02,
    # Final coverage fill — missing silent-pool relics
    "BELT_BUCKLE":      1.02,  # +2 Dex while no potions — proxy
    "CIRCLET":          1.00,  # joke relic, neutral
    "GHOST_SEED":       1.01,  # Strikes/Defends gain Ethereal — small cycling
    "HAND_DRILL":       1.03,  # break-block Vulnerable — small damage proxy
    "MEAT_CLEAVER":     1.02,  # rest-site cook — small healing/resource proxy
    "PRESERVED_FOG":    1.02,  # pickup removes 5 cards + Folly
    "RUNIC_PYRAMID":    1.06,  # no discard hand — extremely strong in STS1, solid proxy
    "SCROLL_BOXES":     1.02,  # lose gold + card pack
    "SWORD_OF_STONE":   1.03,  # transforms after 5 elites
    "VAKUU_CARD_SELECTOR":1.00, # mystery — neutral
    # --- newly added rune/ancient relics (26 relics) ---
    "BOUND_PHYLACTERY": 1.02,  # summon minion proxy
    "BRIMSTONE":        1.03,  # strength gain → direct effect, small bonus
    "CRACKED_CORE":     1.02,  # Channel lightning; damage mult proxy
    "DATA_DISK":        1.02,  # Focus channel; damage mult proxy
    "DEMON_TONGUE":     1.02,  # heal-on-damage; defensive proxy
    "EMOTION_CHIP":     1.02,  # conditional passive; small bonus
    "FUNERARY_MASK":    1.02,  # Soul mechanic; exotic proxy
    "GOLD_PLATED_CABLES":1.02, # extra orb passive; complex mechanic
    "IVORY_TILE":       1.01,  # expensive card energy; small bonus
    "LUNAR_PASTRY":     1.02,  # star gain; exotic proxy
    "METRONOME":        1.02,  # channel threshold; complex proxy
    "MINI_REGENT":      1.02,  # star spend bonus; exotic proxy
    "ORANGE_DOUGH":     1.03,  # free colorless cards on combat start
    "PAPER_PHROG":      1.04,  # vulnerable damage amp; significant damage boost
    "POWER_CELL":       1.02,  # zero-cost cards on combat start
    "RED_SKULL":        1.02,  # strength when low HP; conditional proxy
    "REGALITE":         1.01,  # colorless card creation; small bonus
    "RUINED_HELMET":    1.02,  # first strength gain bonus; small bonus
    "RUNIC_CAPACITOR":  1.03,  # extra orb slots; damage mult proxy
    "SELF_FORMING_CLAY":1.02,  # damage reduction; defensive proxy
    "SYMBIOTIC_VIRUS":  1.02,  # poison channel; damage mult proxy
    "UNDYING_SIGIL":    1.01,  # doom mechanic; defensive/niche proxy
    "VAMBRACE":         1.01,  # first block doubling; already in FIRST_BLOCK_GAIN_DOUBLE
    "VITRUVIAN_MINION": 1.03,  # double damage on minion cards
}

GLOBAL_BLOCK_MULTIPLIERS: dict[str, float] = {
    "STURDY_CLAMP":     1.08,  # block persists
    "ICE_CREAM":        1.05,  # energy conservation
    "RINGING_TRIANGLE": 1.03,
    "DIAMOND_DIADEM":   1.10,
    "TUNGSTEN_ROD":     1.04,
    "PAPER_KRANE":      1.03,
    "FRESNEL_LENS":     1.02,
    "DIVINE_RIGHT":     1.04,
    "BEATING_REMNANT":  1.03,
    "BIG_MUSHROOM":     1.02,
    "LIZARD_TAIL":      1.01,
    "POCKETWATCH":      1.02,
    "GAMBLING_CHIP":    1.02,
    "THE_ABACUS":       1.02,  # also triggers SHUFFLE_TRIGGERS
}

# Flat damage reduction (multiplies incoming damage after block).
# Keep very small — stacking is aggressive.
INCOMING_DAMAGE_REDUCTION: dict[str, float] = {
    "TUNGSTEN_ROD":     0.97,
    "DIAMOND_DIADEM":   0.95,
    "BEATING_REMNANT":  0.95,
    "PAPER_KRANE":      0.96,
    "LIZARD_TAIL":      0.98,
    "THE_BOOT":         1.00,  # neutral but tracked
}


# =============================================================================
# OUT-OF-COMBAT / RUN-SETUP EFFECTS
# =============================================================================
#
# Consumed by full_run.py when a relic enters the owned set.  Combat
# engine never touches these.

# Max HP bonus applied once when the relic is first picked up.
MAX_HP_ON_PICKUP: dict[str, int] = {
    "STRAWBERRY":       7,
    "PEAR":             10,
    "MANGO":            14,
    "LEES_WAFFLE":      7,
    "BIG_MUSHROOM":     20,
    "LOOMING_FRUIT":    31,
    "NUTRITIOUS_OYSTER":11,
    "FAKE_MANGO":       3,
}

# Gold bonus on pickup.
GOLD_ON_PICKUP: dict[str, int] = {
    "OLD_COIN":     300,
    "GOLDEN_PEARL": 150,
    "SIGNET_RING":  999,
    "CURSED_PEARL": 333,
}

# Extra potion slots on pickup.
POTION_SLOTS_ON_PICKUP: dict[str, int] = {
    "POTION_BELT":       2,
    "ALCHEMICAL_COFFER": 4,
    "CAULDRON":          5,
}

# Card reward adjustments on pickup.
CARD_REWARDS_ON_PICKUP: dict[str, int] = {
    "ORRERY":        5,
    "LOST_COFFER":   1,
}

# Flat gold gained per normal enemy kill.
ENEMY_GOLD_BONUS_FLAT: dict[str, int] = {
    "AMETHYST_AUBERGINE": 10,
}

# Multiplicative gold bonus per kill.
ENEMY_GOLD_BONUS_MULT: dict[str, float] = {
    "BOWLER_HAT": 0.20,
}

# Shop price multipliers (0.50 = 50% off).
SHOP_PRICE_MULTIPLIER: dict[str, float] = {
    "MEMBERSHIP_CARD": 0.50,
    "THE_COURIER":     0.80,
}

# Rest-site healing bonus on top of base 30%.
REST_HEAL_BONUS: dict[str, int] = {
    "REGAL_PILLOW": 15,
}

# Shop heal on entry.
SHOP_HEAL: dict[str, int] = {
    "MEAL_TICKET": 15,
}

# ? room heal.
UNKNOWN_ROOM_HEAL: dict[str, int] = {
    "PLANISPHERE": 4,
}

# Max-HP-per-rest-visited (Eternal Feather etc. handled slightly differently).
REST_MAX_HP_GROWTH: dict[str, int] = {
    "STONE_HUMIDIFIER": 5,
}


# =============================================================================
# CONSTANTS used by effects.py to cap runaway multiplier stacking.
# =============================================================================

MAX_STACKED_DAMAGE_MULT = 1.60
MIN_STACKED_DAMAGE_MULT = 0.70
MAX_STACKED_BLOCK_MULT  = 1.50
MIN_STACKED_BLOCK_MULT  = 0.80
MIN_INCOMING_REDUCTION  = 0.70


# =============================================================================
# PUBLIC API — called from combat_engine and effects
# =============================================================================

def _apply_effects(state: "CombatState", effects: dict, elite: bool = False,
                   boss: bool = False) -> None:
    """Apply a single effects-dict to the state.

    ``effects`` may contain any of the keys documented at the top of
    this module.  Unknown keys are silently ignored so new entries can
    be added to the data tables without touching this dispatcher.
    """
    from .effects import calculate_block_gain  # local import to avoid cycles

    p = state.player

    # --- Start-of-combat / turn-start primitives ---
    if block := effects.get("block"):
        p.block += block
    if thorns := effects.get("thorns"):
        p.powers["Thorns"] = p.powers.get("Thorns", 0) + thorns
    if strength := effects.get("strength"):
        p.powers["Strength"] = p.powers.get("Strength", 0) + strength
    if dexterity := effects.get("dexterity"):
        p.powers["Dexterity"] = p.powers.get("Dexterity", 0) + dexterity
    if vigor := effects.get("vigor"):
        # Approximate Vigor (+N to next Attack) as Strength for simplicity.
        p.powers["Strength"] = p.powers.get("Strength", 0) + max(1, vigor // 4)
    if heal := effects.get("heal"):
        p.hp = min(p.hp + heal, p.max_hp)
    if energy := effects.get("energy"):
        p.energy += energy
    if draw := effects.get("draw"):
        from .effects import draw_cards
        draw_cards(state, draw)
    if dmg := effects.get("damage_all"):
        _damage_all(state, dmg)
    if dmg := effects.get("damage_random"):
        _damage_random(state, dmg)
    if vuln := effects.get("vulnerable_all"):
        for e in state.enemies:
            if e.is_alive:
                e.powers["Vulnerable"] = e.powers.get("Vulnerable", 0) + vuln
    if weak := effects.get("weak_all"):
        for e in state.enemies:
            if e.is_alive:
                e.powers["Weak"] = e.powers.get("Weak", 0) + weak
    if poison := effects.get("poison_all"):
        for e in state.enemies:
            if e.is_alive:
                e.powers["Poison"] = e.powers.get("Poison", 0) + poison
    if plating := effects.get("plating"):
        # Plating → modelled as persistent block
        p.block += plating
    if shivs := effects.get("shivs_to_hand"):
        from .card_registry import _make_shiv
        for _ in range(shivs):
            if len(p.hand) < 10:
                p.hand.append(_make_shiv())
    if sd := effects.get("self_damage"):
        p.hp = max(1, p.hp - sd)

    # --- Turn-start conditional / this-turn-only ---
    if eff := effects.get("strength_this_turn"):
        p.powers["Strength"] = p.powers.get("Strength", 0) + eff
    if eff := effects.get("dexterity_this_turn"):
        p.powers["Dexterity"] = p.powers.get("Dexterity", 0) + eff
    if eff := effects.get("block_per_hand_card"):
        p.block += eff * len(p.hand)


def apply_start_of_combat(state: "CombatState", is_elite: bool = False,
                          is_boss: bool = False) -> None:
    """Fire every start-of-combat relic effect on ``state``."""
    relics = state.relics

    for rid in relics:
        if rid in START_OF_COMBAT:
            _apply_effects(state, START_OF_COMBAT[rid])
        if is_elite and rid in ELITE_ONLY_START:
            _apply_effects(state, ELITE_ONLY_START[rid])
        if is_boss and rid in BOSS_ONLY_START:
            _apply_effects(state, BOSS_ONLY_START[rid])

    # Akabeko proxy: add +1 Strength this combat
    if "AKABEKO" in relics:
        state.player.powers["Strength"] = (
            state.player.powers.get("Strength", 0) + AKABEKO_STRENGTH_PROXY)

    # Strike Dummy: legacy "gain 1 Str per Strike in deck" kept as a
    # start-of-combat bonus so the old behaviour isn't lost.
    if "STRIKE_DUMMY" in relics:
        strikes = sum(1 for c in state.player.draw_pile
                      if "Strike" in c.name)
        if strikes > 0:
            state.player.powers["Strength"] = (
                state.player.powers.get("Strength", 0) + strikes)


def apply_turn_start(state: "CombatState") -> None:
    """Fire every turn-start relic effect on ``state``.  Uses ``state.turn``
    which must already be incremented before this call."""
    relics = state.relics
    turn = state.turn

    for rid in relics:
        entries = TURN_START.get(rid)
        if not entries:
            continue
        for eff in entries:
            # Turn gating
            if (t := eff.get("turn")) is not None and turn != t:
                continue
            if (t := eff.get("turn_min")) is not None and turn < t:
                continue
            if (t := eff.get("turn_max")) is not None and turn > t:
                continue
            # Marker-handled specials are no-ops here
            if "_marker" in eff:
                continue
            _apply_effects(state, eff)


def apply_card_play(state: "CombatState", card: "Card") -> None:
    """Fire per-card-play triggers after a card has been played.

    The counter state lives in ``state.player.powers`` under private
    keys (prefixed with an underscore) so it automatically clears at
    end of combat along with the rest of the powers dict.
    """
    from .constants import CardType
    relics = state.relics
    powers = state.player.powers

    for rid, rule in CARD_PLAY_TRIGGERS.items():
        if rid not in relics:
            continue
        if not _matches_card(card, rule.get("match", "any"), state):
            continue

        effect = rule.get("effect", {})

        # damage_bonus-style entries are handled via GLOBAL_DAMAGE_MULTIPLIERS
        # so they don't need a counter — skip them here.
        if "damage_bonus" in effect:
            continue

        every = rule.get("every", 1)
        scope = rule.get("scope", "always")

        if every == 1 and scope == "always":
            # Fire every time
            _apply_effects(state, effect)
            continue

        key = f"_{rid}_count"
        count = powers.get(key, 0) + 1
        if count >= every:
            _apply_effects(state, effect)
            count = 0
        powers[key] = count


def _matches_card(card: "Card", match: str, state: "CombatState") -> bool:
    """Predicate dispatch for CARD_PLAY_TRIGGERS["match"] values."""
    from .constants import CardType
    from .combat_engine import effective_cost

    if match == "any":
        return True
    if match == "attack":
        return card.card_type == CardType.ATTACK
    if match == "skill":
        return card.card_type == CardType.SKILL
    if match == "power":
        return card.card_type == CardType.POWER
    if match == "upgraded_attack":
        return card.card_type == CardType.ATTACK and card.upgraded
    if match == "strike_in_name":
        return "Strike" in card.name
    if match == "expensive":
        try:
            return effective_cost(state, card) >= 2
        except Exception:
            return card.cost >= 2
    if match == "shiv":
        return card.id == "SHIV" or card.name == "Shiv"
    if match == "trifecta":
        # Rainbow Ring: first Attack+Skill+Power each turn.  Tracked via
        # three boolean flags in powers.
        kind = card.card_type
        flag_map = {
            CardType.ATTACK: "_rainbow_attack",
            CardType.SKILL:  "_rainbow_skill",
            CardType.POWER:  "_rainbow_power",
        }
        flag = flag_map.get(kind)
        if not flag:
            return False
        if state.player.powers.get(flag, 0):
            return False
        state.player.powers[flag] = 1
        if (state.player.powers.get("_rainbow_attack", 0)
                and state.player.powers.get("_rainbow_skill", 0)
                and state.player.powers.get("_rainbow_power", 0)):
            return True
        return False
    return False


def apply_discard(state: "CombatState", count: int = 1) -> None:
    """Fire DISCARD_TRIGGERS for each discarded card."""
    relics = state.relics
    for rid, eff in DISCARD_TRIGGERS.items():
        if rid in relics:
            for _ in range(count):
                _apply_effects(state, eff)


def apply_shuffle(state: "CombatState") -> None:
    """Fire SHUFFLE_TRIGGERS when the draw pile reshuffles."""
    relics = state.relics
    for rid, eff in SHUFFLE_TRIGGERS.items():
        if rid in relics:
            _apply_effects(state, eff)


def apply_exhaust(state: "CombatState") -> None:
    """Fire EXHAUST_TRIGGERS when a card is exhausted."""
    relics = state.relics
    for rid, eff in EXHAUST_TRIGGERS.items():
        if rid in relics:
            _apply_effects(state, eff)

    # Joss Paper: every 5 exhausts, draw 1
    if "JOSS_PAPER" in relics:
        key = "_joss_count"
        c = state.player.powers.get(key, 0) + 1
        if c >= 5:
            from .effects import draw_cards
            draw_cards(state, 1)
            c = 0
        state.player.powers[key] = c


def apply_first_hp_loss(state: "CombatState") -> None:
    """Fire FIRST_HP_LOSS_TRIGGERS once per combat."""
    relics = state.relics
    powers = state.player.powers
    for rid, eff in FIRST_HP_LOSS_TRIGGERS.items():
        if rid in relics:
            flag = f"_{rid}_fired"
            if not powers.get(flag):
                powers[flag] = 1
                _apply_effects(state, eff)


def apply_enemy_death(state: "CombatState") -> None:
    """Fire ENEMY_DEATH_TRIGGERS whenever an enemy dies."""
    relics = state.relics
    for rid, eff in ENEMY_DEATH_TRIGGERS.items():
        if rid in relics:
            _apply_effects(state, eff)


def apply_end_of_turn(state: "CombatState") -> None:
    """Fire end-of-turn relic effects.  Mirrors combat_engine.end_turn."""
    relics = state.relics
    p = state.player

    for rid, eff in END_OF_TURN.items():
        if rid not in relics:
            continue
        if "_marker" in eff:
            continue
        if "block_per_hand_card" in eff:
            p.block += eff["block_per_hand_card"] * len(p.hand)
        if "if_no_block" in eff and p.block == 0:
            _apply_effects(state, eff["if_no_block"])
        if "if_no_attacks" in eff and state.attacks_played_this_turn == 0:
            _apply_effects(state, eff["if_no_attacks"])
        if (cond := eff.get("if_block_at_least")) is not None:
            threshold, sub = cond
            if p.block >= threshold:
                _apply_effects(state, sub)
        if "if_empty_hand" in eff and not p.hand:
            _apply_effects(state, eff["if_empty_hand"])
        if (cond := eff.get("if_played_at_most")) is not None:
            threshold, sub = cond
            if state.cards_played_this_turn <= threshold:
                if "_marker" in sub:
                    p.powers["_pocketwatch_next_draw"] = 3


def apply_end_of_combat(state: "CombatState") -> None:
    """Fire end-of-combat relic effects (healing etc.)."""
    relics = state.relics
    p = state.player
    for rid, eff in END_OF_COMBAT.items():
        if rid not in relics:
            continue
        if heal := eff.get("heal"):
            p.hp = min(p.hp + heal, p.max_hp)
        if heal := eff.get("heal_if_below_half"):
            if p.hp <= p.max_hp // 2:
                p.hp = min(p.hp + heal, p.max_hp)
        if mhp := eff.get("max_hp"):
            p.max_hp += mhp
            p.hp = min(p.hp + mhp, p.max_hp)


# =============================================================================
# MULTIPLIER AGGREGATION
# =============================================================================

def get_damage_multiplier(relics: Iterable[str]) -> float:
    """Return the aggregate damage multiplier for a set of owned relics."""
    if not relics:
        return 1.0
    mult = 1.0
    for rid in relics:
        m = GLOBAL_DAMAGE_MULTIPLIERS.get(rid)
        if m is not None:
            mult *= m
    return max(MIN_STACKED_DAMAGE_MULT,
               min(MAX_STACKED_DAMAGE_MULT, mult))


def get_block_multiplier(relics: Iterable[str]) -> float:
    """Return the aggregate block multiplier for a set of owned relics."""
    if not relics:
        return 1.0
    mult = 1.0
    for rid in relics:
        m = GLOBAL_BLOCK_MULTIPLIERS.get(rid)
        if m is not None:
            mult *= m
    return max(MIN_STACKED_BLOCK_MULT,
               min(MAX_STACKED_BLOCK_MULT, mult))


def get_incoming_damage_reduction(relics: Iterable[str]) -> float:
    """Return the aggregate incoming-damage multiplier.  Lower = safer."""
    if not relics:
        return 1.0
    mult = 1.0
    for rid in relics:
        m = INCOMING_DAMAGE_REDUCTION.get(rid)
        if m is not None:
            mult *= m
    return max(MIN_INCOMING_REDUCTION, min(1.0, mult))


# =============================================================================
# LOCAL HELPERS
# =============================================================================

def _damage_all(state: "CombatState", amount: int) -> None:
    """AoE damage that respects enemy block, used by the effects tables."""
    for e in state.enemies:
        if not e.is_alive:
            continue
        dmg = amount
        if e.block > 0:
            if dmg >= e.block:
                dmg -= e.block
                e.block = 0
            else:
                e.block -= dmg
                dmg = 0
        e.hp -= dmg


def _damage_random(state: "CombatState", amount: int) -> None:
    """Damage the first living enemy (deterministic for solver)."""
    for e in state.enemies:
        if e.is_alive:
            dmg = amount
            if e.block > 0:
                if dmg >= e.block:
                    dmg -= e.block
                    e.block = 0
                else:
                    e.block -= dmg
                    dmg = 0
            e.hp -= dmg
            return


# =============================================================================
# SIMULATED RELIC SET
# =============================================================================
#
# The union of every relic ID that has *any* behavioural effect in this
# module.  Used by full_run.py to build the in-game drop pool from the
# set of relics the training loop can actually reward the agent for.

def simulated_relic_ids() -> set[str]:
    """All relic IDs that this module models in some way (combat or OOC)."""
    ids: set[str] = set()
    ids.update(START_OF_COMBAT.keys())
    ids.update(ELITE_ONLY_START.keys())
    ids.update(BOSS_ONLY_START.keys())
    ids.update(TURN_START.keys())
    ids.update(CARD_PLAY_TRIGGERS.keys())
    ids.update(DISCARD_TRIGGERS.keys())
    ids.update(SHUFFLE_TRIGGERS.keys())
    ids.update(EXHAUST_TRIGGERS.keys())
    ids.update(FIRST_HP_LOSS_TRIGGERS.keys())
    ids.update(FIRST_BLOCK_GAIN_DOUBLE)
    ids.update(END_OF_TURN.keys())
    ids.update(END_OF_COMBAT.keys())
    ids.update(GLOBAL_DAMAGE_MULTIPLIERS.keys())
    ids.update(GLOBAL_BLOCK_MULTIPLIERS.keys())
    ids.update(INCOMING_DAMAGE_REDUCTION.keys())
    ids.update(MAX_HP_ON_PICKUP.keys())
    ids.update(GOLD_ON_PICKUP.keys())
    ids.update(POTION_SLOTS_ON_PICKUP.keys())
    ids.update(CARD_REWARDS_ON_PICKUP.keys())
    ids.update(ENEMY_GOLD_BONUS_FLAT.keys())
    ids.update(ENEMY_GOLD_BONUS_MULT.keys())
    ids.update(SHOP_PRICE_MULTIPLIER.keys())
    ids.update(REST_HEAL_BONUS.keys())
    ids.update(SHOP_HEAL.keys())
    ids.update(UNKNOWN_ROOM_HEAL.keys())
    ids.update(REST_MAX_HP_GROWTH.keys())
    return ids
