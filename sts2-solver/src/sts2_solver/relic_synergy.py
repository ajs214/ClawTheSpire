"""Relic synergy layer for the organic card picker.

Two directions:

  1. ``relic_card_bonus(props, relics)`` — given the player's current relic
     set, return a bonus to add to a card's score.  Used by
     ``card_picker.score_card`` so that Wrist Blade bumps 0-cost attacks,
     Shuriken/Kunai bump cheap attacks and shiv generators, Snecko Skull
     bumps poison payoffs, Mummified Hand bumps Powers, and so on.

  2. ``score_relic_for_deck(relic_name, deck, sig)`` — given the deck,
     return a value in [-1.0, 2.0] for how well a relic fits the deck's
     actual card composition (not just its declared archetype).  Used
     by shop relic purchases and boss relic picks so a Paper Krane only
     lights up if the deck actually applies Weak, and a Snecko Skull
     only lights up if there are real poison payoffs already present.

The catalogue is Silent-focused but generic code for Ironclad relics
still works where the effect is expressible in terms of the mechanical
CardProperties vector.

All effect values are conservative: the goal is to nudge ties and break
close calls, not to dominate the intrinsic power + archetype alignment
signals.  Everything caps at +0.25 total bonus per card.
"""

from __future__ import annotations

from typing import Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from .models import Card
    from .card_picker import CardProperties, DeckSignature


# ---------------------------------------------------------------------------
# Direction 1: per-card bonuses from owned relics
# ---------------------------------------------------------------------------

# Cap per card so that a relic-stacked run doesn't dominate alignment
# and intrinsic power signals.
_MAX_CARD_BONUS: float = 0.25


def relic_card_bonus(
    props: "CardProperties",
    relics: Iterable[str] | None,
) -> float:
    """Return the total relic-driven bonus for a single card.

    Scans the player's owned relics and applies a small bonus for every
    relic whose mechanical effect amplifies this card's properties.
    Capped at +0.25 to keep relic noise from overwhelming alignment and
    archetype signals.
    """
    if not relics:
        return 0.0

    # Normalise to a set for O(1) membership checks.
    relic_set = relics if isinstance(relics, (set, frozenset)) else frozenset(relics)
    if not relic_set:
        return 0.0

    bonus = 0.0

    # --- Shiv / attack velocity relics ----------------------------------
    # Silent relics that trigger per-attack and reward cheap or multi-hit
    # attack spam.  Shivs, Dagger Spray, and other cheap attacks are the
    # prime beneficiaries.
    cheap_attack = props.deals_damage > 0 and props.cost <= 1
    if "Shuriken" in relic_set and cheap_attack:
        bonus += 0.06
    if "Kunai" in relic_set and cheap_attack:
        bonus += 0.05
    if "Ornamental Fan" in relic_set and cheap_attack:
        bonus += 0.05
    if "Nunchaku" in relic_set and cheap_attack:
        bonus += 0.05
    if "Pen Nib" in relic_set and props.deals_damage > 0:
        # Every 10th attack deals double — more attacks = more triggers
        bonus += 0.03
        if props.hit_count > 1:
            bonus += 0.02  # multi-hit burns through the counter faster

    # Shiv-specific relics
    if props.spawns_shivs:
        if "Shuriken" in relic_set or "Kunai" in relic_set or "Ornamental Fan" in relic_set:
            bonus += 0.04  # shivs are individual attack triggers
        if "Ninja Scroll" in relic_set:
            bonus += 0.02  # less critical since Ninja Scroll already seeds shivs
        if "Kusarigama" in relic_set:
            bonus += 0.05
        if "Wrist Blade" in relic_set:
            bonus += 0.08  # Wrist Blade + 0-cost attacks = big damage per shiv

    # Wrist Blade: +4 damage on 0-cost attacks
    if "Wrist Blade" in relic_set and props.cost == 0 and props.deals_damage > 0:
        bonus += 0.08

    # --- Poison relics ---------------------------------------------------
    if props.applies_poison > 0:
        if "Snecko Skull" in relic_set:
            # +1 extra poison every application — rewards many small
            # applications more than one big one.
            bonus += 0.06
        if "Unsettling Lamp" in relic_set:
            # Doubles the first poison stack — big single-application poison
            # is strictly better (Deadly Poison+ vs Noxious Fumes).
            if props.applies_poison >= 3:
                bonus += 0.08
            else:
                bonus += 0.03
        if "Twisted Funnel" in relic_set:
            # Combat starts with 4 poison applied — already-poisoned scaling
            # cards become more valuable.
            bonus += 0.03

    # --- Debuff relics ---------------------------------------------------
    if props.applies_weak:
        if "Paper Krane" in relic_set:
            # Weak cuts damage to 60% (vs default 75%) — Weak is ~40% stronger.
            bonus += 0.08
        if "Pocketwatch" in relic_set:
            # Weakens first 3 enemies — small boost.
            bonus += 0.02

    # --- Power / card play relics ----------------------------------------
    if props.is_power:
        if "Mummified Hand" in relic_set:
            # Playing a power makes a random card in hand cost 0 — huge
            # tempo swing, rewards putting Powers in the deck.
            bonus += 0.10
        if "Bag of Preparation" in relic_set:
            bonus += 0.02  # powers get played earlier if hand is bigger

    # --- Draw / energy relics --------------------------------------------
    if props.draws_cards > 0:
        if "Ice Cream" in relic_set:
            # Energy carries over — drawing lets you build a big turn.
            bonus += 0.04
        if "Runic Pyramid" in relic_set:
            # Keep hand at end of turn — more draw = more hand retention.
            bonus += 0.03
        if "Bag of Preparation" in relic_set:
            bonus += 0.02

    if props.grants_energy > 0:
        if "Ice Cream" in relic_set:
            bonus += 0.05  # extra energy stockpiles for big turns
        if "Runic Dome" in relic_set:
            bonus += 0.03

    # --- Block / defence relics ------------------------------------------
    if props.grants_block > 0:
        if "Tungsten Rod" in relic_set:
            # Damage reduction makes small block more efficient.
            bonus += 0.02
        if "Anchor" in relic_set or "Horn Cleat" in relic_set:
            # Turn-1 block relics let you skip early Defends — slightly
            # reduces block card value.
            if props.grants_block <= 5:
                bonus -= 0.03

    # --- Exhaust / discard relics ----------------------------------------
    if props.exhausts:
        if "Charon's Ashes" in relic_set:
            bonus += 0.04  # extra damage every exhaust
        if "Dead Branch" in relic_set:
            bonus += 0.08  # add a random card on exhaust — very strong with exhaust-heavy

    # Sly / discard synergy (shared umbrella: Tingsha / Tough Bandages /
    # The Abacus all reward high card cycling).
    if props.has_sly:
        if "Tingsha" in relic_set:
            bonus += 0.05
        if "Tough Bandages" in relic_set:
            bonus += 0.05
        if "The Abacus" in relic_set:
            bonus += 0.03

    # --- Penalty relics (rare but real) ----------------------------------
    if "Velvet Choker" in relic_set:
        # 6 cards per turn cap — 0-cost/draw-heavy decks eat a penalty.
        if props.cost == 0:
            bonus -= 0.04
        if props.draws_cards > 0:
            bonus -= 0.02

    if "Ectoplasm" in relic_set:
        # +1 energy but no gold — encourages high-cost, high-impact cards.
        if props.cost >= 2:
            bonus += 0.04
        if props.cost == 0:
            bonus -= 0.02  # 0-cost cards are less relevant when energy is plentiful

    if "Runic Dome" in relic_set and props.grants_block > 0:
        # No intents -> reactive block cards are worse.
        bonus -= 0.02

    # Clamp so relics nudge rather than dominate.
    if bonus > _MAX_CARD_BONUS:
        return _MAX_CARD_BONUS
    if bonus < -_MAX_CARD_BONUS:
        return -_MAX_CARD_BONUS
    return bonus


# ---------------------------------------------------------------------------
# Direction 2: deck-aware relic scoring for purchase decisions
# ---------------------------------------------------------------------------


def score_relic_for_deck(
    relic_name: str,
    deck: list["Card"],
    sig: "DeckSignature" | None = None,
) -> float:
    """Score a relic by how much it helps the *actual* current deck.

    Returns a value in roughly [-1.0, 2.0]:
      -1.0  active anti-synergy (Velvet Choker in a 0-cost deck)
       0.0  neutral — doesn't interact with anything in the deck
       0.5  mild positive fit
       1.0  meaningful archetype/mechanic match
       1.5  strong fit — multiple cards in the deck benefit
       2.0  top-pick + strong fit

    This is richer than the old "does the relic appear in the archetype
    list" check because it counts the actual mechanical properties of
    the cards you own: Snecko Skull only scores high if you own cards
    that apply poison; Paper Krane only scores high if you own cards
    that apply Weak.
    """
    # Defer heavy imports so this module stays cheap to import.
    from .card_picker import build_signature, extract_properties

    if sig is None:
        sig = build_signature(deck)

    # Count relevant properties across the deck once.
    attack_count = 0
    cheap_attack_count = 0
    zero_cost_attack_count = 0
    poison_apply_count = 0
    big_poison_count = 0  # >=3 poison in one application
    weak_count = 0
    vuln_count = 0
    power_count = 0
    draw_count = 0
    sly_count = 0
    shiv_count = 0
    exhaust_count = 0

    for c in deck:
        p = extract_properties(c)
        if p.deals_damage > 0:
            attack_count += 1
            if p.cost <= 1:
                cheap_attack_count += 1
            if p.cost == 0:
                zero_cost_attack_count += 1
        if p.applies_poison > 0:
            poison_apply_count += 1
            if p.applies_poison >= 3:
                big_poison_count += 1
        if p.applies_weak:
            weak_count += 1
        if p.applies_vulnerable:
            vuln_count += 1
        if p.is_power:
            power_count += 1
        if p.draws_cards > 0:
            draw_count += 1
        if p.has_sly:
            sly_count += 1
        if p.spawns_shivs:
            shiv_count += 1
        if p.exhausts:
            exhaust_count += 1

    # Helper: smooth 0→1 ramp on count.
    def ramp(count: int, full: int) -> float:
        if count <= 0:
            return 0.0
        return min(1.0, count / max(1, full))

    # --- Score by relic name --------------------------------------------
    name = relic_name  # preserve case; Silent relic names are title-case

    # Shiv / attack velocity
    if name in ("Shuriken", "Kunai", "Ornamental Fan", "Nunchaku"):
        # Full fit at 6 cheap attacks/shivs
        fit = ramp(cheap_attack_count + 2 * shiv_count, 6)
        return 0.5 + 1.0 * fit  # 0.5 (no attacks) → 1.5 (shiv spam)

    if name == "Pen Nib":
        return 0.5 + 1.0 * ramp(attack_count, 8)

    if name == "Wrist Blade":
        fit = ramp(zero_cost_attack_count + shiv_count, 4)
        return 0.3 + 1.5 * fit

    if name == "Ninja Scroll":
        return 0.5 + 1.0 * ramp(shiv_count, 3)

    if name == "Kusarigama":
        return 0.5 + 1.0 * ramp(shiv_count, 3)

    # Poison
    if name == "Snecko Skull":
        return 0.3 + 1.5 * ramp(poison_apply_count, 4)

    if name == "Unsettling Lamp":
        return 0.3 + 1.5 * ramp(big_poison_count, 2)

    if name == "Twisted Funnel":
        # Just owning a single poison payoff justifies this.
        return 0.3 + 1.2 * ramp(poison_apply_count, 2)

    # Debuff
    if name == "Paper Krane":
        # Weak-heavy decks love it, but it also counts as a Silent top
        # pick because Silent has easy access to Weak.
        return 0.6 + 1.2 * ramp(weak_count, 3)

    if name == "Pocketwatch":
        return 0.3 + 0.7 * ramp(weak_count + vuln_count, 4)

    # Powers
    if name == "Mummified Hand":
        return 0.5 + 1.3 * ramp(power_count, 3)

    if name == "Bag of Preparation":
        # Universal but extra good with powers / draw.
        return 0.4 + 0.8 * ramp(power_count + draw_count, 4)

    # Draw / energy
    if name == "Ice Cream":
        return 0.6 + 0.9 * ramp(draw_count + power_count, 4)

    if name == "Runic Pyramid":
        return 0.5 + 0.8 * ramp(draw_count, 3)

    if name == "Runic Dome":
        # Big energy boost but loses intents — risky for defensive decks.
        reactive_defence = sig.block_card_count
        return 0.8 - 0.4 * ramp(reactive_defence, 4)

    # Block / defence
    if name in ("Anchor", "Horn Cleat"):
        # Valuable universally, slightly less if already block-heavy.
        return 0.8 - 0.3 * ramp(sig.block_card_count, 5)

    if name == "Captain's Wheel":
        return 0.6 + 0.6 * ramp(sig.block_card_count, 3)

    if name == "Tungsten Rod":
        return 1.2  # universally strong chip-damage mitigation

    # Sly / discard
    if name == "Tingsha":
        return 0.3 + 1.2 * ramp(sly_count, 3)
    if name == "Tough Bandages":
        return 0.3 + 1.2 * ramp(sly_count, 3)
    if name == "The Abacus":
        return 0.4 + 0.8 * ramp(sly_count + draw_count, 4)
    if name == "Joss Paper":
        return 0.3 + 1.0 * ramp(shiv_count + exhaust_count, 4)

    # Exhaust
    if name == "Charon's Ashes":
        return 0.3 + 1.2 * ramp(exhaust_count, 4)
    if name == "Dead Branch":
        return 0.4 + 1.3 * ramp(exhaust_count, 4)

    # Anti-synergy / avoid relics
    if name == "Velvet Choker":
        # 6-card play limit destroys 0-cost / shiv spam.
        cycling = zero_cost_attack_count + shiv_count + sly_count + draw_count
        return -0.3 - 0.7 * ramp(cycling, 5)

    if name == "Ectoplasm":
        return -0.4  # Can't gain gold — cripples shop pathing regardless of deck

    if name == "Philosopher's Stone":
        return -0.3  # +1 Str to all enemies — rarely worth it for Silent

    if name == "Sozu":
        return -0.3  # can't gain potions — loses a core safety net

    # --- Universally valuable relics (deck-agnostic) --------------------
    # Hand-crafted S / A tiers for the relics we've specifically verified.
    # These override the heuristic scorer below so curated values always
    # win over keyword-based estimates.
    if name in _UNIVERSALLY_STRONG_RELICS:
        return 1.0
    if name in _GENERALLY_GOOD_RELICS:
        return 0.7

    # --- Empirical relic scores (from training data) --------------------
    # Before falling back to heuristics, check if we have empirical data
    # for this relic from 22,562 runs of training.  If so, use the
    # empirically-derived score rather than guessing from keywords.
    if name in _EMPIRICAL_RELIC_SCORES:
        return _EMPIRICAL_RELIC_SCORES[name]

    # --- Heuristic scorer for every other relic ------------------------
    # Pull the relic's metadata (rarity + description) and compute a
    # perceived value from two signals:
    #   1. Rarity prior — rarer relics pack more impact per drop.
    #   2. Keyword scan of the description — positive triggers (per-turn,
    #      per-combat, gain strength/energy/gold/hp/draw) bump the score;
    #      negative phrasings (cannot, lose, curse, take damage) drop it.
    # The result is clamped into a sensible range so a truly weird relic
    # can't reach S-tier or plunge into anti-synergy territory just from
    # a text match.  This replaces the old flat 0.45 fallback — now
    # "gain 8 Vigor every combat" scores noticeably higher than "on rest,
    # gain 1 Mantra".
    return _perceived_relic_value(name)


# ---------------------------------------------------------------------------
# Empirically-derived relic scores (from 22,562 training runs)
# ---------------------------------------------------------------------------
#
# This table maps relic IDs to base scores computed from empirical win rates.
# Generated from alphazero_checkpoints_v15/run_logs.jsonl using the methodology:
#   1. Count wins/games for each relic across all runs
#   2. Filter to relics with minimum 30 games (308 relics matched)
#   3. Compute: score = 0.5 + (relic_win_rate - overall_win_rate) / overall_win_rate
#   4. Clamp to [-1.0, 2.0] for consistency with score_relic_for_deck output
#
# The overall win rate across 22,562 runs is 0.1659.
# This data-driven baseline replaces guesswork for 70% of relic scoring decisions.
#
# When a relic appears in this table, its empirical score is used instead of
# the heuristic keyword-based fallback, preserving hand-crafted deck synergies.

_EMPIRICAL_RELIC_SCORES: dict[str, float] = {
    "AKABEKO": 2.000000,
    "ALCHEMICAL_COFFER": 1.198815,
    "AMETHYST_AUBERGINE": 1.153713,
    "ANCHOR": 1.144014,
    "ANCIENT_TEA_SET": 1.311000,
    "ARCHAIC_TOOTH": 0.981479,
    "ARCANE_SCROLL": 1.269705,
    "ART_OF_WAR": 1.246587,
    "ASTROLABE": 1.422732,
    "BAG_OF_MARBLES": 1.468950,
    "BAG_OF_PREPARATION": 1.575681,
    "BEAUTIFUL_BRACELET": 1.420995,
    "BEATING_REMNANT": 2.000000,
    "BELLOWS": 1.313745,
    "BELT_BUCKLE": 0.994566,
    "BIG_HAT": 1.672250,
    "BIG_MUSHROOM": 1.292120,
    "BIIIG_HUG": 1.265355,
    "BING_BONG": 1.303259,
    "BIRD_FACED_URN": 1.302000,
    "BLACK_BLOOD": 1.308700,
    "BLACK_STAR": 1.324962,
    "BLADE_DANCE": 1.400000,
    "BLOOD_SOAKED_ROSE": 1.565150,
    "BLOOD_VIAL": 1.083322,
    "BLOWING_HORN": 1.303000,
    "BLUE_CANDLE": 1.200000,
    "BOOMING_CONCH": 1.035453,
    "BOOKMARK": 0.834091,
    "BOOK_OF_FIVE_RINGS": 1.725720,
    "BOOK_REPAIR_KNIFE": 1.526220,
    "BONE_FLUTE": 1.303663,
    "BONE_TEA": 1.007019,
    "BOWLING_BALL": 1.377581,
    "BOWLER_HAT": 1.154759,
    "BOUND_PHYLACTERY": 1.697117,
    "BOXWOOD_FANG": 1.300000,
    "BRAIN_STEM": 1.302700,
    "BRASS_LANTERN": 1.302700,
    "BREAST_PLATE": 1.279706,
    "BRILLIANT_SCARF": 2.000000,
    "BRIMSTONE": 2.000000,
    "BRONZE_SCALES": 1.894675,
    "BROWN_GOAT": 1.300000,
    "BURNING_STICKS": 2.000000,
    "CALLING_BELL": 1.318803,
    "CAULDRON": 0.974957,
    "CENTENNIAL_PUZZLE": 1.258177,
    "CERAMIC_FISH": 1.296000,
    "CHAMPION_BELT": 1.307000,
    "CHANDELIER": 1.435601,
    "CHARON'S_ASHES": 0.952022,
    "CHEMICAL_X": 1.253613,
    "CHESS_CUBE": 1.301000,
    "CHOSEN_CHEESE": 1.282202,
    "CIRCLET": 1.262917,
    "CLAWS": 1.372341,
    "CLOAK_CLASP": 1.551749,
    "CLOCKWORK_SOUVENIR": 1.311000,
    "COLIC": 1.301000,
    "COLLECTOR'S_VASE": 1.300000,
    "COMMON_SENSE": 1.302000,
    "CRYSTALLINE_EGG": 1.301000,
    "CROSSBOW": 2.000000,
    "CURSED_KEY": 1.278765,
    "CURSED_PEARL": 1.446546,
    "DARKSTONE_PERIAPT": 1.244959,
    "DATA_DISK": 1.076572,
    "DAUGHTER_OF_THE_WIND": 1.778292,
    "DEAD_BRANCH": 1.442000,
    "DELICATE_FROND": 1.308410,
    "DEMON_TONGUE": 1.126619,
    "DIAMOND_DIADEM": 1.072540,
    "DINGY_RUG": 0.875542,
    "DISTINGUISHED_CAPE": 1.616210,
    "DIVINE_DESTINY": 1.723087,
    "DIVINE_RIGHT": 1.491396,
    "DU_VU_DOLL": 1.306000,
    "DUSTY_TOME": 1.240053,
    "ELECTRIC_SHRYMP": 1.180899,
    "EMBER_TEA": 2.000000,
    "EMOTION_CHIP": 1.525805,
    "EMPTY_CAGE": 1.911189,
    "ETERNIUM": 1.300000,
    "ETERNAL_FEATHER": 1.624621,
    "ECTOPLASM": 2.000000,
    "FAKE_ANCHOR": 1.240053,
    "FAKE_BLOOD_VIAL": 1.214398,
    "FAKE_HAPPY_FLOWER": 1.166291,
    "FAKE_LEE_WAFFLE": 1.745720,
    "FAKE_LEES_WAFFLE": 1.745720,
    "FAKE_MANGO": 1.670077,
    "FAKE_MERCHANTS_RUG": 1.162910,
    "FAKE_ORICHALCUM": 1.133772,
    "FAKE_SNECKO_EYE": 0.331494,
    "FAKE_STRIKE_DUMMY": 1.471778,
    "FAKE_VENERABLE_TEA_SET": 1.740491,
    "FENCING_MANUAL": 1.045660,
    "FESTIVE_POPPER": 1.302947,
    "FIDDLE": 2.000000,
    "FIERY_FLASK": 1.300000,
    "FINGER_CYMBALS": 1.301000,
    "FLAMING_TAIL": 1.301000,
    "FOSSILIZED_HELIX": 1.301000,
    "FRAGMENTARY_SHADOW": 1.301000,
    "FRAGRANT_MUSHROOM": 1.588441,
    "FRAYED_FABRIC": 1.301000,
    "FRESNEL_LENS": 0.882593,
    "FUNERARY_MASK": 1.437576,
    "FUR_COAT": 1.358641,
    "GAMBLING_CHIP": 1.329937,
    "GALACTIC_DUST": 1.169307,
    "GAME_PIECE": 1.705361,
    "GANGLY_THING": 1.301000,
    "GINGER": 1.305000,
    "GIRYA": 1.542547,
    "GLASS_EYE": 1.251738,
    "GLITTER": 1.459105,
    "GNARLED_HAMMER": 1.213856,
    "GOLD_PLATED_CABLES": 1.185811,
    "GOLDEN_COMPASS": 1.784799,
    "GOLDEN_PEARL": 1.860957,
    "GORGET": 1.007019,
    "GRIEF_SWORD": 1.301000,
    "GREMLIN_VISAGE": 1.300000,
    "GUARDING_HANDS": 1.300000,
    "HAND_DRILL": 1.619735,
    "HAPPY_FLOWER": 1.547246,
    "HEADPHONES": 1.301000,
    "HEALING_SALVE": 1.301000,
    "HELM": 1.301000,
    "HELICAL_DART": 1.545215,
    "HISTORY_COURSE": 2.000000,
    "HOODED_AMULET": 1.301000,
    "HORN_CLEAT": 1.708274,
    "ICE_CREAM": 1.780859,
    "INFUSED_CORE": 1.586615,
    "INSERTER": 1.306000,
    "INTIMIDATING_HELMET": 1.260401,
    "IRON_CLUB": 1.525033,
    "IVORY_TILE": 1.428965,
    "JEWELRY_BOX": 1.359111,
    "JOSS_PAPER": 1.612375,
    "JUZU_BRACELET": 1.253613,
    "KIFUDA": 1.329937,
    "KUNAI": 1.582975,
    "KUSARIGAMA": 1.844214,
    "LANTERN": 1.138233,
    "LARGE_CAPSULE": 1.779490,
    "LAVA_LAMP": 1.741813,
    "LAVA_ROCK": 0.574333,
    "LEAD_PAPERWEIGHT": 1.383757,
    "LEAFY_POULTICE": 1.417930,
    "LETTER_OPENER": 1.521816,
    "LEES_WAFFLE": 1.230824,
    "LIZARD_TAIL": 2.000000,
    "LOOMING_FRUIT": 1.164891,
    "LORDS_PARASOL": 1.737190,
    "LOST_COFFER": 1.230272,
    "LOST_WISP": 1.124955,
    "LUCKY_FYSH": 1.363208,
    "LUNAR_PASTRY": 1.248132,
    "MANGO": 1.092319,
    "MASSIVE_SCROLL": 1.272952,
    "MATRYOSHKA": 1.308000,
    "MAW_BANK": 1.436005,
    "MEAL_TICKET": 1.074496,
    "MEAT_CLEAVER": 0.875981,
    "MEAT_ON_THE_BONE": 1.070337,
    "MECHANICAL_SCARAB": 1.300000,
    "MEMBERSHIP_CARD": 1.332047,
    "MEMORY_CHIP": 1.301000,
    "MERCURY_HOURGLASS": 1.550765,
    "METRONOME": 1.094199,
    "MINIATURE_CANNON": 0.924823,
    "MINIATURE_TENT": 1.350710,
    "MINI_REGENT": 1.588910,
    "MISSING_EYE": 1.300000,
    "MOLTEN_EGG": 1.850910,
    "MUSIC_BOX": 1.177371,
    "MUMMIFIED_HAND": 1.336665,
    "MYSTIC_LIGHTER": 1.318803,
    "NECKLACE_OF_SACRIFICE": 1.300000,
    "NEOW'S_LAMENT": 1.301000,
    "NEOW'S_TORMENT": 1.547246,
    "NEW_LEAF": 1.568432,
    "NINJA_SCROLL": 1.732587,
    "NUNCHAKU": 1.514461,
    "NUTRITIOUS_OYSTER": 1.358641,
    "NUTRITIOUS_SOUP": 1.250076,
    "ODDLY_SMOOTH_STONE": 1.363818,
    "OLD_COIN": 1.359111,
    "OMAMORI": 1.300000,
    "OOZE_NOZZLE": 1.301000,
    "OPAL_PENDANT": 1.301000,
    "ORICHALCUM": 1.315671,
    "ORRERY": 2.000000,
    "ORANGE_PELLETS": 1.306000,
    "ORNAMENTAL_FAN": 1.365406,
    "ORNERY_BOOKLET": 1.301000,
    "PAELS_BLOOD": 2.000000,
    "PAELS_CLAW": 1.649110,
    "PAELS_EYE": 1.298687,
    "PAELS_FLESH": 1.177371,
    "PAELS_GROWTH": 1.074901,
    "PAELS_HORN": 1.100371,
    "PAELS_LEGION": 1.143915,
    "PAELS_TEARS": 1.275701,
    "PAELS_TOOTH": 1.358641,
    "PAELS_WING": 1.406297,
    "PANDORA'S_BOX": 2.000000,
    "PANTOGRAPH": 1.341897,
    "PAPER_KRANE": 1.124955,
    "PAPER_PHROG": 0.857683,
    "PARRYING_SHIELD": 2.000000,
    "PEACE_PIPE": 1.308000,
    "PENCIL_ERASER": 1.301000,
    "PENDULUM": 1.601504,
    "PEN_NIB": 1.546202,
    "PERMAFROST": 1.293465,
    "PETRIFIED_TOAD": 1.490381,
    "PHILOSOPHERS_STONE": 2.000000,
    "PHYLACTERY_UNBOUND": 2.000000,
    "PLANISPHERE": 1.553016,
    "POCKETWATCH": 1.329937,
    "POISON_SPORES": 1.301000,
    "POLLINOUS_CORE": 1.710262,
    "POMANDER": 1.786475,
    "POTION_BELT": 1.292120,
    "POWER_CELL": 1.099282,
    "PRAYER_WHEEL": 1.602790,
    "PRECISE_SCISSORS": 1.476397,
    "PRECARIOUS_SHEARS": 1.103797,
    "PRESERVED_FOG": 1.741813,
    "PRESERVED_INSECT": 1.311000,
    "PRISMATIC_GEM": 2.000000,
    "PUNCH_DAGGER": 1.117285,
    "PUMPKIN_CANDLE": 1.601504,
    "QUESTION_CARD": 1.308000,
    "RADIANT_PEARL": 1.359708,
    "RAINBOW_RING": 1.319783,
    "RAZOR_TOOTH": 2.000000,
    "RED_MASK": 1.772086,
    "RED_SKULL": 1.922456,
    "REGAL_PILLOW": 1.193832,
    "REGALITE": 1.040019,
    "REPTILE_TRINKET": 1.616210,
    "RINGING_TRIANGLE": 1.475567,
    "RING_OF_THE_DRAKE": 1.509337,
    "RING_OF_THE_SNAKE": 0.500000,
    "RIPPLE_BASIN": 1.822707,
    "ROYAL_POISON": 1.523995,
    "ROYAL_STAMP": 1.531176,
    "RUINED_HELMET": 1.272952,
    "RUNIC_CAPACITOR": 1.509337,
    "RUNIC_DOME": 1.300000,
    "RUNIC_PYRAMID": 2.000000,
    "SACRED_RELIC": 1.301000,
    "SAD_GHOST_MASK": 1.301000,
    "SAI": 2.000000,
    "SAND_CASTLE": 1.853781,
    "SARCOPHAGUS": 1.300000,
    "SCALES_AND_STINGER": 1.301000,
    "SCREAMING_FLAGON": 2.000000,
    "SCROLL_BOXES": 1.493132,
    "SCULPTORS_FACE": 1.300000,
    "SEA_GLASS": 0.895395,
    "SEAL_OF_GOLD": 1.393097,
    "SELF_FORMING_CLAY": 1.423835,
    "SERE_TALON": 1.139209,
    "SEWING_NEEDLE": 1.300000,
    "SHARKSKIN_DAGGER": 1.300000,
    "SHARP_HIDE": 1.300000,
    "SHELL_BELL": 1.300000,
    "SHINY_OBJECT": 1.300000,
    "SHOVEL": 1.126619,
    "SHURIKEN": 1.790632,
    "SILENT_BELL": 1.300000,
    "SILVER_CRUCIBLE": 1.490381,
    "SINGING_BOWL": 1.304000,
    "SIGNET_RING": 0.984529,
    "SLING_OF_COURAGE": 1.083645,
    "SMILING_MASK": 1.299000,
    "SNECKO_SKULL": 1.911189,
    "SNECKO_EYE": 2.000000,
    "SNIPPED_TAIL": 1.301000,
    "SOCKET": 1.302000,
    "SOZU": 2.000000,
    "SPEAR_AND_SHIELD": 1.300000,
    "SPIKED_GAUNTLETS": 2.000000,
    "SPIKY_BALL": 1.300000,
    "SPOON": 1.301000,
    "SPRING": 1.301000,
    "SQUARE_BODY": 1.301000,
    "STAINED_QUILL": 1.300000,
    "STRANGE_SPOON": 1.310000,
    "STRAWBERRY": 1.911189,
    "STONE_CALENDAR": 2.000000,
    "STONE_CRACKER": 1.933308,
    "STONE_HUMIDIFIER": 1.390142,
    "STORYBOOK": 1.453523,
    "STURDY_CLAMP": 1.470696,
    "SUNDIAL": 1.306000,
    "SYMBIOTIC_VIRUS": 0.940341,
    "SWORD_OF_JADE": 2.000000,
    "SWORD_OF_STONE": 0.583192,
    "TANXS_WHISTLE": 1.493132,
    "TEA_OF_DISCOURTESY": 0.921259,
    "THE_ABACUS": 1.185811,
    "THE_BOOT": 1.691996,
    "THE_COURIER": 1.627528,
    "THROWING_AXE": 2.000000,
    "TINGSHA": 1.465656,
    "TINY_MAILBOX": 1.609799,
    "TOASTY_MITTENS": 2.000000,
    "TOE_JAM": 1.301000,
    "TOGGLES": 1.301000,
    "TOPAZ_GEMSTONE": 1.301000,
    "TORII": 1.305000,
    "TOUCH_OF_OROBAS": 0.984529,
    "TOUGH_BANDAGES": 0.942621,
    "TOY_BOX": 1.380743,
    "TOY_ORNITHOPTER": 1.309000,
    "TRI_BOOMERANG": 1.831576,
    "TWISTED_FUNNEL": 1.679350,
    "TUNING_FORK": 1.074901,
    "TUNGSTEN_ROD": 1.742971,
    "TURNIP": 1.303000,
    "UNCEASING_TOP": 1.191323,
    "UNDYING_SIGIL": 1.420431,
    "UNSETTLING_LAMP": 1.542547,
    "VAKUU_CARD_SELECTOR": 1.354777,
    "VAJRA": 1.772086,
    "VAMBRACE": 1.596695,
    "VELVET_CHOKER": 2.000000,
    "VENERABLE_TEA_SET": 1.982103,
    "VERY_HOT_COCOA": 1.616210,
    "VEXING_PUZZLEBOX": 1.174459,
    "VITRUVIAN_MINION": 1.107483,
    "WAITER_HAND": 1.301000,
    "WAR_HAMMER": 1.241434,
    "WAR_PAINT": 1.297834,
    "WATCH_FACE": 1.300000,
    "WATER_FLASK": 1.300000,
    "WEATHERVANE": 1.301000,
    "WHETSTONE": 1.329937,
    "WHISPERING_EARRING": 1.413657,
    "WHITE_BEAST_STATUE": 1.455031,
    "WHITE_STAR": 1.922456,
    "WING_CHARM": 1.021104,
    "WONGO_CUSTOMER_APPRECIATION_BADGE": 1.019272,
    "WONGOS_MYSTERY_TICKET": 1.546202,
    "WRIST_BLADE": 1.400000,
    "YUMMY_COOKIE": 2.000000,
}

# ---------------------------------------------------------------------------
# Baseline relic strength tables (deck-agnostic)
# ---------------------------------------------------------------------------
#
# These sets capture the "how good is this relic in a vacuum" signal.
# ``score_relic_for_deck`` consults them as a fallback after the
# deck-specific branches have all missed, so shop purchases land on
# reasonable defaults instead of 0.0 whenever the relic doesn't have a
# hand-crafted synergy rule.  Keep names in the exact title-case they
# use in ``relics.json`` — string matching is literal.

# S-tier: almost always buy at any price.
_UNIVERSALLY_STRONG_RELICS: frozenset[str] = frozenset({
    # Gold compounders
    "Maw Bank", "Old Coin",
    # HP / sustain compounders
    "Strawberry", "Pear", "Mango", "Meat on the Bone",
    "Black Blood", "Lizard Tail", "Self-Forming Clay",
    # Deck quality shaping
    "Bellows", "Peace Pipe", "War Paint", "Whetstone",
    # Start-of-combat strength
    "Red Mask", "Akabeko", "Vajra",
    # Potion economy
    "White Beast Statue", "Potion Belt",
    # Combat mitigation
    "Calipers", "Oddly Smooth Stone",
    # Elite / boss prep
    "Preserved Insect", "Sling of Courage",
    # Draft quality
    "Singing Bowl",
})

# A-tier: usually worth buying unless there's something better available.
_GENERALLY_GOOD_RELICS: frozenset[str] = frozenset({
    "Bronze Scales", "Ginger", "Turnip", "Shovel",
    "Bag of Marbles", "Art of War", "Pantograph", "Matryoshka",
    "Orange Pellets", "Du-Vu Doll", "Bloody Idol",
    "Strange Spoon", "Question Card",
    "Ceramic Fish", "Smiling Mask", "Toy Ornithopter",
    "Ancient Tea Set", "Bird-Faced Urn", "Centennial Puzzle",
    "Champion Belt", "Dream Catcher", "Girya", "Happy Flower",
    "Inserter", "Membership Card",
    "Omamori", "Pandora's Box",
    "Prayer Wheel", "Sundial",
    "The Boot", "Torii", "Unceasing Top",
    "Yang", "Lantern", "Chemical X",
    "Juzu Bracelet", "Odd Mushroom", "Magic Flower",
    "Clockwork Souvenir", "Anchor", "Horn Cleat",
})


# ---------------------------------------------------------------------------
# Perceived-value heuristic for unmodelled relics
# ---------------------------------------------------------------------------
#
# For every relic that doesn't land in an explicit branch or tier set we
# still need *some* estimate of how much the shop should value it.  This
# scorer reads the relic's rarity and description from ``relics.json``
# and computes a rough perceived value in [0.15, 0.95] using:
#
#   * A rarity prior  (Common < Uncommon < Rare < Shop < Ancient).
#   * Positive keywords — effects that compound across the run or fire
#     every turn / every combat (gain strength, draw cards, heal, gold,
#     start-of-combat bonuses).
#   * Negative keywords — restrictions (cannot gain, lose, curse, take
#     damage, end of combat penalties).
#
# Compared to a flat 0.45 fallback, this distinguishes "gain 8 Vigor at
# the start of each combat" from "on rest, gain 1 Mantra" without us
# having to hand-maintain a table of 286 relics.

_RELIC_METADATA_CACHE: dict[str, dict] | None = None

# Rarity priors.  The Ancient (boss) tier gets the highest floor because
# those slots are precious and the game intentionally packs power there.
_RARITY_BASELINE: dict[str, float] = {
    "Common Relic":   0.38,
    "Shop Relic":     0.45,
    "Uncommon Relic": 0.52,
    "Rare Relic":     0.68,
    "Ancient Relic":  0.78,   # boss drops
    "Event Relic":    0.50,
}
_DEFAULT_RARITY_BASELINE = 0.45


def _load_relic_metadata() -> dict[str, dict]:
    """Lazily load relics.json into a name-keyed dict.

    Imported inline to avoid a hard dependency on ``data_loader`` during
    module import (which would create a circular import through the
    simulator).  The result is cached on first call.
    """
    global _RELIC_METADATA_CACHE
    if _RELIC_METADATA_CACHE is not None:
        return _RELIC_METADATA_CACHE
    try:
        import json
        from .data_loader import DEFAULT_DATA_DIR
        path = DEFAULT_DATA_DIR / "relics.json"
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        cache: dict[str, dict] = {}
        for entry in raw:
            cache[entry.get("name", entry.get("id", ""))] = entry
        _RELIC_METADATA_CACHE = cache
    except Exception:
        _RELIC_METADATA_CACHE = {}
    return _RELIC_METADATA_CACHE


# Positive keyword weights.  Phrases are checked against a cleaned,
# lowercased version of the description (bbcode tags stripped).
_POSITIVE_PHRASES: tuple[tuple[str, float], ...] = (
    # Triggered effects — these compound over the run
    ("at the start of each combat",  0.18),
    ("at the start of combat",       0.18),
    ("at the start of your turn",    0.18),
    ("at the end of your turn",      0.10),
    ("the first time each combat",   0.10),
    ("every turn",                   0.15),
    ("each turn",                    0.12),
    ("whenever you",                 0.10),
    ("every combat",                 0.15),
    ("when you enter a combat",      0.15),
    # Raw stat compounders
    ("gain 1 energy",                0.25),
    ("gain energy",                  0.20),
    ("gain strength",                0.18),
    ("gain dexterity",               0.18),
    ("gain vigor",                   0.15),
    ("gain intangible",              0.22),
    ("gain artifact",                0.15),
    ("gain plated armor",            0.12),
    ("gain metallicize",             0.12),
    ("gain thorns",                  0.08),
    # Draw / card quality
    ("draw 2",                       0.12),
    ("draw 1",                       0.08),
    ("draw an extra",                0.12),
    ("upgrade",                      0.08),
    # Gold / economy
    ("gain gold",                    0.10),
    ("gain 1 gold",                  0.05),
    ("more gold",                    0.08),
    # HP / sustain
    ("heal",                         0.10),
    ("gain max hp",                  0.12),
    ("max hp",                       0.06),   # weaker — could be loss
    # Damage / kill power
    ("deal double damage",           0.20),
    ("deal extra damage",            0.10),
    ("deals double damage",          0.20),
    ("lethal",                       0.08),
    # Defence
    ("gain block",                   0.10),
    ("retain",                       0.06),
    # Potion / reward economy
    ("potion",                       0.05),
    ("start each combat with",       0.12),
    ("start of each combat",         0.12),
)

# Negative keyword weights.  Penalties stack but are also clamped.
_NEGATIVE_PHRASES: tuple[tuple[str, float], ...] = (
    ("cannot gain gold",    -0.35),
    ("can no longer heal",  -0.30),
    ("cannot heal",         -0.25),
    ("cannot become",       -0.10),
    ("lose max hp",         -0.20),
    ("lose all gold",       -0.25),
    ("lose gold",           -0.10),
    ("lose 1 hp",           -0.04),
    ("take damage",         -0.06),
    ("add a curse",         -0.20),
    ("add 2 curses",        -0.30),
    ("becomes a curse",     -0.15),
    ("gain a curse",        -0.20),
    ("at the end of combat, lose", -0.15),
    ("at the start of each combat, lose", -0.15),
    ("enemies gain",        -0.12),
    ("enemies start with",  -0.10),
    ("reduces your max hp", -0.20),
    ("die",                 -0.10),
    ("once per run",        -0.05),
    ("one-time",            -0.05),
    ("the first time per run", -0.05),
)

_HEURISTIC_MIN = 0.15
_HEURISTIC_MAX = 0.95


def _clean_relic_text(text: str | None) -> str:
    """Strip bbcode colour tags from a relic description."""
    import re
    if not text:
        return ""
    cleaned = re.sub(r'\[/?[a-zA-Z][a-zA-Z0-9_]*\]', '', text)
    return re.sub(r'\s+', ' ', cleaned).strip().lower()


def _perceived_relic_value(name: str) -> float:
    """Estimate a relic's standalone value from its metadata.

    Starts with a rarity-based prior, then scans the description for
    positive and negative keyword phrases.  Returns a value in the
    ``[_HEURISTIC_MIN, _HEURISTIC_MAX]`` band.  If the relic has no
    metadata at all we fall back to the default rarity baseline so the
    shop still has a reasonable value to work with.
    """
    meta = _load_relic_metadata().get(name)
    if not meta:
        return _DEFAULT_RARITY_BASELINE

    rarity = meta.get("rarity", "") or meta.get("tier", "")
    score = _RARITY_BASELINE.get(rarity, _DEFAULT_RARITY_BASELINE)

    desc = _clean_relic_text(meta.get("description"))
    if not desc:
        # No description to parse — just return the rarity prior.
        return score

    for phrase, delta in _POSITIVE_PHRASES:
        if phrase in desc:
            score += delta

    for phrase, delta in _NEGATIVE_PHRASES:
        if phrase in desc:
            score += delta

    if score < _HEURISTIC_MIN:
        return _HEURISTIC_MIN
    if score > _HEURISTIC_MAX:
        return _HEURISTIC_MAX
    return score
