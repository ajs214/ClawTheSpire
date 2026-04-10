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

    # Unknown relic — neutral.
    return 0.0
