"""Organic card picker for Slay the Spire 2 (Silent).

Design principles (maxims):
  1. Early picks are free — take the best card offered. Each pick narrows
     future choices. Penalise going off-archetype proportionally to commitment.
  2. Archetypes are discovered from card *properties* (applies poison, spawns
     shivs, has Sly keyword, draws cards, grants block) — never hardcoded lists.
  3. Balance (offense/defense/draw) matters late, not early. The first picks
     should maximise power and synergy; balance gaps become relevant once the
     deck has an identity.

Architecture:
  - DeckSignature: a vector of mechanical property counts for a deck
  - CardSignature: same vector for a single card
  - Alignment score: how well a card extends the deck's direction
  - Rule-based scorer: implements the maxims
  - pick_card(): public entry point used by simulator, AlphaZero self-play,
    and the live in-game advisor
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from .models import Card
from .constants import CardType, TargetType


# ---------------------------------------------------------------------------
# Card mechanical properties — derived from Card data, never from names
# ---------------------------------------------------------------------------

@dataclass
class CardProperties:
    """Mechanical property vector for a single card."""
    applies_poison: float = 0.0     # total poison applied
    spawns_shivs: bool = False      # spawns SHIV cards
    has_sly: bool = False           # has Sly keyword
    draws_cards: int = 0            # cards_draw value
    grants_block: float = 0.0       # block value
    deals_damage: float = 0.0       # damage value
    grants_energy: int = 0          # energy_gain
    is_power: bool = False          # CardType.POWER
    is_aoe: bool = False            # targets ALL_ENEMIES
    hit_count: int = 1              # multi-hit
    cost: int = 0                   # energy cost
    exhausts: bool = False          # exhaust keyword
    applies_weak: bool = False      # applies Weak
    applies_vulnerable: bool = False  # applies Vulnerable
    grants_dexterity: bool = False  # applies Dexterity
    grants_strength: bool = False   # applies Strength
    grants_intangible: bool = False  # applies Intangible/Wraith Form


# Var keys that indicate poison-related effects (detected from card.vars)
_POISON_VAR_KEYS = frozenset({
    "poisonperturn", "poisonpower", "poison",
    "envenom", "envenompower",
    "accelerant",
    "corrosivewave",
})

# Var keys that indicate intangible effects
_INTANGIBLE_VAR_KEYS = frozenset({
    "intangiblepower", "intangible",
})


def extract_properties(card: Card) -> CardProperties:
    """Extract mechanical properties from a Card object."""
    props = CardProperties()
    props.cost = card.cost if card.cost >= 0 else 0
    props.deals_damage = card.damage or 0
    props.grants_block = card.block or 0
    props.grants_energy = card.energy_gain
    props.is_power = card.card_type == CardType.POWER
    props.is_aoe = card.target == TargetType.ALL_ENEMIES
    props.hit_count = card.hit_count
    props.exhausts = card.exhausts
    props.has_sly = "Sly" in card.keywords
    props.spawns_shivs = "SHIV" in card.spawns_cards

    # Card draw: if the card spawns shivs, the "draw" is actually shiv
    # generation, not real deck draw. Only count non-shiv draw.
    if props.spawns_shivs:
        props.draws_cards = 0  # Shiv "draw" handled by spawns_shivs
    else:
        props.draws_cards = card.cards_draw

    for pname, pval in card.powers_applied:
        pname_lower = pname.lower()
        if pname_lower == "poison":
            props.applies_poison = pval
        elif pname_lower == "weak":
            props.applies_weak = True
        elif pname_lower == "vulnerable":
            props.applies_vulnerable = True
        elif pname_lower == "dexterity":
            props.grants_dexterity = True
        elif pname_lower == "strength":
            props.grants_strength = True
        elif pname_lower in ("intangible", "wraith form"):
            props.grants_intangible = True

    # Detect effects from card vars (covers cards whose effects are
    # implemented in the combat engine and not in powers_applied).
    # This replaces hardcoded name lists — the vars are intrinsic card data.
    card_vars = getattr(card, 'vars', {}) or {}
    card_var_keys = {k.lower() for k in card_vars}

    # Poison detection from vars
    if props.applies_poison == 0 and card_var_keys & _POISON_VAR_KEYS:
        # Use the actual var value for the best estimate
        for vk in ("PoisonPerTurn", "PoisonPower", "Poison", "Envenom",
                    "EnvenomPower", "Accelerant", "CorrosiveWave"):
            if vk in card_vars:
                props.applies_poison = max(props.applies_poison, card_vars[vk])

    # Intangible detection from vars
    if not props.grants_intangible and card_var_keys & _INTANGIBLE_VAR_KEYS:
        props.grants_intangible = True

    # Last-resort fallback: description text (for any future cards
    # that might not have vars either)
    desc = getattr(card, 'description', '') or ''
    desc_lower = desc.lower()
    if props.applies_poison == 0 and '[gold]poison[/gold]' in desc_lower:
        props.applies_poison = 2  # Conservative estimate
    if not props.grants_intangible and '[gold]intangible[/gold]' in desc_lower:
        props.grants_intangible = True

    return props


# ---------------------------------------------------------------------------
# Deck signature — emergent archetype identity from property counts
# ---------------------------------------------------------------------------

@dataclass
class DeckSignature:
    """Aggregate property vector for an entire deck.

    The archetype is never declared — it emerges from the relative magnitudes
    of these counts. A deck with high poison_total and low shiv_count is a
    poison deck. A deck with both is mixed (and probably weaker for it).
    """
    size: int = 0
    poison_total: float = 0.0       # sum of poison applied across all cards
    poison_card_count: int = 0      # number of cards that apply poison
    shiv_card_count: int = 0        # number of cards that spawn shivs
    sly_card_count: int = 0         # number of cards with Sly keyword
    draw_total: int = 0             # total cards_draw across deck
    draw_card_count: int = 0        # number of cards that draw
    block_total: float = 0.0        # total block across deck
    block_card_count: int = 0       # number of cards with block
    damage_total: float = 0.0       # total damage across deck
    power_count: int = 0            # number of Power cards
    aoe_count: int = 0              # number of AoE cards
    avg_cost: float = 0.0           # average energy cost
    zero_cost_count: int = 0        # number of 0-cost cards
    energy_gain_count: int = 0      # cards that give energy
    weak_count: int = 0             # cards that apply Weak
    dexterity_count: int = 0        # cards that give Dexterity
    intangible_count: int = 0       # cards that grant Intangible

    # Non-starter cards only (excludes Strike/Defend)
    picked_count: int = 0           # cards added beyond starter deck

    @property
    def dominant_archetype(self) -> str:
        """The emergent archetype, or 'undecided' if none dominates.

        V7 tuning (from V6 boss-log analysis): "undecided" had the HIGHEST
        win rate of any archetype (10.2% vs shiv 8.8%, sly 4.9%, poison
        4.1%, mixed 3.7%), and commitment was *inversely* correlated with
        winning.  Pool simulation showed only 15-22% of Act 1 runs even
        *see* 5+ archetype cards, so committing on 2 actively hurts.

        New gate: require 3+ archetype cards AND separation of 2 from the
        second-best count.  This keeps the picker in "best card available"
        mode through the entire Act 1 unless the run happens to be lucky.
        """
        scores = {
            "poison": self.poison_card_count,
            "shiv": self.shiv_card_count,
            "sly": self.sly_card_count,
        }
        best = max(scores, key=scores.get)
        best_count = scores[best]
        # V7: require 3+ cards (not 2) before leaving undecided
        if best_count < 3:
            return "undecided"
        # V7: require separation of 2 (not 1) — a 3/2/0 split is still mixed
        second = sorted(scores.values(), reverse=True)[1]
        if best_count - second < 2:
            return "mixed"
        return best

    @property
    def archetype_commitment(self) -> float:
        """0.0 = no direction, 1.0 = fully committed.

        V7 tuning: denominator floor raised from 3 to 5.  The V6 boss-log
        data showed decks with fewer than 5 archetype payoffs couldn't
        actually execute their game plan and lost more than undecided
        starter-heavy decks.  "Full commitment" now means 5+ archetype
        cards, matching the empirical threshold where an archetype deck
        starts to function as intended.
        """
        scores = [self.poison_card_count, self.shiv_card_count, self.sly_card_count]
        best = max(scores)
        if best == 0 or self.picked_count == 0:
            return 0.0
        return min(1.0, best / max(5, self.picked_count))

    @property
    def defense_ratio(self) -> float:
        """Fraction of deck that provides block."""
        return self.block_card_count / max(1, self.size)

    @property
    def draw_ratio(self) -> float:
        """Fraction of deck that draws cards."""
        return self.draw_card_count / max(1, self.size)


_STARTER_NAMES = frozenset({"strike", "defend", "survivor", "neutralize"})


# V7: Empirical Act-1 premium neutrals, derived from the V6 boss-log.
# These are the cards with the highest win-lift (ratio of freq-in-wins to
# freq-in-losses) in the 3,651-fight V6 dataset, filtered to cards that
# appeared in >=5 wins so the sample is meaningful.  All of these are
# *neutral* generalists — no archetype commitment — and they're the
# cards the previous picker was systematically rejecting when it
# committed to shiv/poison/sly too early.
#
# Bonus is additive to score_card and only applies in Act 1 (floor < 15; FIX 1: was 17)
# where the picker has the least information and the data is cleanest.
# Kept modest (+0.10) so it nudges rather than dominates.
_ACT1_PREMIUM_NEUTRALS = frozenset({
    # Massive lifts (>2x in wins)
    "grand finale",         # 23.4x lift
    "anticipate",           # 2.27x lift, 64% of wins vs 28% of losses
    "piercing wail",        # 3.25x lift
    "restlessness",         # 2.58x lift
    "jackpot",              # 3.91x lift
    # Strong lifts (1.4-2x)
    "footwork",             # 1.46x lift
    "dramatic entrance",    # 1.45x lift
    "sucker punch",         # 1.30x lift
    "slice",                # 1.38x lift
    "leading strike",       # 1.22x lift
    "well laid plans",      # 1.67x lift
    "expertise",            # 1.61x lift
    "seeker strike",        # 1.64x lift
    "hidden daggers",       # 1.91x lift
    "volley",               # 1.85x lift
    "adrenaline",           # 2.39x lift
    "snakebite",            # 1.22x lift
    "finesse",              # 1.21x lift
})


def _is_premium_neutral(card: Card) -> bool:
    """Is this card in the empirically-derived Act 1 premium neutral set?"""
    return card.name.strip().lower() in _ACT1_PREMIUM_NEUTRALS


def build_signature(deck: list[Card]) -> DeckSignature:
    """Build a deck signature from mechanical properties."""
    sig = DeckSignature(size=len(deck))
    costs = []

    for card in deck:
        props = extract_properties(card)

        if props.applies_poison > 0:
            sig.poison_total += props.applies_poison
            sig.poison_card_count += 1
        if props.spawns_shivs:
            sig.shiv_card_count += 1
        if props.has_sly:
            sig.sly_card_count += 1
        if props.draws_cards > 0:
            sig.draw_total += props.draws_cards
            sig.draw_card_count += 1
        if props.grants_block > 0:
            sig.block_total += props.grants_block
            sig.block_card_count += 1
        if props.deals_damage > 0:
            sig.damage_total += props.deals_damage
        if props.is_power:
            sig.power_count += 1
        if props.is_aoe:
            sig.aoe_count += 1
        if props.cost == 0:
            sig.zero_cost_count += 1
        if props.grants_energy > 0:
            sig.energy_gain_count += 1
        if props.applies_weak:
            sig.weak_count += 1
        if props.grants_dexterity:
            sig.dexterity_count += 1
        if props.grants_intangible:
            sig.intangible_count += 1

        costs.append(props.cost)

        # Track non-starter picks
        if card.name.lower() not in _STARTER_NAMES:
            sig.picked_count += 1

    sig.avg_cost = sum(costs) / max(1, len(costs))
    return sig


# ---------------------------------------------------------------------------
# Archetype classification (for reporting)
# ---------------------------------------------------------------------------

@dataclass
class ArchetypeReport:
    """Classification of a deck for reporting purposes."""
    archetype: str          # "poison", "shiv", "sly", "mixed", "undecided"
    commitment: float       # 0.0–1.0
    poison_cards: int
    shiv_cards: int
    sly_cards: int
    block_cards: int
    draw_cards: int
    deck_size: int
    picked_count: int


def classify_deck(deck: list[Card]) -> ArchetypeReport:
    """Classify a deck's archetype for reporting."""
    sig = build_signature(deck)
    return ArchetypeReport(
        archetype=sig.dominant_archetype,
        commitment=sig.archetype_commitment,
        poison_cards=sig.poison_card_count,
        shiv_cards=sig.shiv_card_count,
        sly_cards=sig.sly_card_count,
        block_cards=sig.block_card_count,
        draw_cards=sig.draw_card_count,
        deck_size=sig.size,
        picked_count=sig.picked_count,
    )


# ---------------------------------------------------------------------------
# Rule-based scorer — implements the three maxims
# ---------------------------------------------------------------------------

def _card_power_score(card: Card, props: CardProperties) -> float:
    """Intrinsic card quality score (0.0–1.0).

    This is the "how good is this card in a vacuum" component.
    Used primarily for early picks when the deck has no direction yet.
    """
    score = 0.0

    # Powers are inherently valuable (permanent effects)
    if props.is_power:
        score += 0.25

    # Damage efficiency
    if props.deals_damage > 0:
        dmg_per_cost = props.deals_damage / max(1, props.cost)
        score += min(0.2, dmg_per_cost * 0.02)
        # Multi-hit bonus (scales with Strength)
        if props.hit_count > 1:
            score += 0.1

    # Block efficiency
    if props.grants_block > 0:
        block_per_cost = props.grants_block / max(1, props.cost)
        score += min(0.15, block_per_cost * 0.02)

    # Draw is always valuable (deck cycling)
    if props.draws_cards > 0:
        score += props.draws_cards * 0.1

    # Energy gain is powerful
    if props.grants_energy > 0:
        score += props.grants_energy * 0.15

    # Zero cost = always playable
    if props.cost == 0:
        score += 0.1

    # AoE helps in multi-enemy fights
    if props.is_aoe:
        score += 0.08

    # Debuffs are strong
    if props.applies_weak:
        score += 0.08
    if props.applies_vulnerable:
        score += 0.06

    # Scaling effects
    if props.grants_dexterity:
        score += 0.2
    if props.grants_strength:
        score += 0.15
    if props.grants_intangible:
        score += 0.3  # Intangible is the strongest effect in the game

    # Poison has delayed but powerful value
    # n poison = n*(n+1)/2 total damage over time — very efficient
    # Account for repeat mechanics (e.g. Bouncing Flask: 3 poison × 3 repeats = 9)
    poison_amount = props.applies_poison
    card_vars = getattr(card, 'vars', {}) or {}
    repeat = card_vars.get('Repeat', 1)
    effective_poison = poison_amount * repeat
    if effective_poison > 0:
        score += min(0.30, effective_poison * 0.03)

    # Shiv generation: each shiv is ~4 damage + scales with Strength/Accuracy
    # The number of shivs generated is stored in the original cards_draw before
    # we zeroed it out (spawns_shivs cards have draw = shiv count in raw data)
    if props.spawns_shivs:
        shiv_count = card.cards_draw or 1  # raw cards_draw = shiv count
        score += min(0.30, 0.08 + shiv_count * 0.06)  # 1 shiv=0.14, 3 shivs=0.26

    # Sly is the strongest keyword — free card plays
    if props.has_sly:
        score += 0.12

    return min(1.0, score)


def _alignment_score(
    card: Card,
    props: CardProperties,
    sig: DeckSignature,
) -> float:
    """How well does this card align with the deck's emerging direction?

    Returns -0.10 (off-archetype) to +0.5 (perfect fit).
    Magnitude scales with deck commitment — early on, everything is ~0.

    V7 retuning (against 3,651 boss fights from V6):
      * Commitment gate raised 0.20 -> 0.35 so the alignment logic only
        fires when the deck is *genuinely* committed under the new
        5-card denominator floor.  Most Act 1 runs never trip it.
      * Off-archetype penalty softened -0.18 -> -0.10 because the V6
        data showed undecided decks had the best WR and the penalty was
        rejecting neutral premium cards (Anticipate, Piercing Wail,
        Footwork) that were the *actual* top win-lift cards.
      * HIGH-POWER NEUTRAL EXEMPTION: if the card's intrinsic power
        score is >= 0.35 (the same "strong card" threshold used for the
        duplicate penalty), the off-archetype penalty is suppressed
        entirely.  A premium generalist always beats a weak archetype
        pick regardless of direction.

    Tuned against 306-boss-fight data (April 2026):
      * Block added to universal support and its magnitude raised
        (0.10 -> 0.20) because block+draw cards drive wins in the data.
      * Cross-synergies expanded to include debuff/scaling pairings.
    """
    commitment = sig.archetype_commitment
    if commitment < 0.35:
        return 0.0  # No direction yet — all cards equally valid

    archetype = sig.dominant_archetype
    if archetype == "undecided" or archetype == "mixed":
        return 0.0

    # --- Core in-archetype fit (by mechanical properties) ---
    is_in_archetype = False
    if archetype == "poison" and props.applies_poison > 0:
        is_in_archetype = True
    elif archetype == "shiv" and props.spawns_shivs:
        is_in_archetype = True
    elif archetype == "sly" and props.has_sly:
        is_in_archetype = True

    # --- Cross-archetype synergies (secondary fit) ---
    # These are cards that aren't the core payoff but meaningfully
    # enable or amplify the archetype's plan.
    is_cross_synergy = False
    if archetype == "poison":
        # Debuffs buy time for poison damage to scale
        if props.applies_weak or props.applies_vulnerable:
            is_cross_synergy = True
        # Sly lets us rip through setup cards to get poison online
        if props.has_sly:
            is_cross_synergy = True
    elif archetype == "shiv":
        # Strength/Dex scale every shiv
        if props.grants_dexterity or props.grants_strength:
            is_cross_synergy = True
        # Vulnerable amplifies every shiv hit
        if props.applies_vulnerable:
            is_cross_synergy = True
        # Sly = free shiv plays
        if props.has_sly:
            is_cross_synergy = True
    elif archetype == "sly":
        # Shiv generators feed Sly with cheap plays
        if props.spawns_shivs:
            is_cross_synergy = True
        # Debuffs extend the runway when we're drawing thin
        if props.applies_weak or props.applies_vulnerable:
            is_cross_synergy = True

    # --- Universal support (helps any archetype) ---
    # Block added: the data shows block cards are the #1 win driver,
    # and they were previously eating the off-archetype penalty in
    # committed decks.
    is_universal_support = (
        props.draws_cards > 0
        or props.grants_energy > 0
        or props.grants_block >= 8  # real block, not incidental riders
    )

    if is_in_archetype:
        return 0.5 * commitment    # Max +0.5 at full commitment
    elif is_cross_synergy:
        return 0.30 * commitment   # Cross-synergy meaningfully lifted
    elif is_universal_support:
        return 0.20 * commitment   # Draw/energy/block always helps

    # V7: high-power neutral exemption — a strong generalist always
    # beats a weak archetype pick regardless of direction.  The V6 data
    # showed the top-lift cards (Anticipate, Piercing Wail, Footwork,
    # Restlessness, Jackpot) were ALL neutral.  Don't penalise them.
    power = _card_power_score(card, props)
    if power >= _STRONG_CARD_THRESHOLD:
        return 0.0

    return -0.10 * commitment  # V7: softened further (-0.18 -> -0.10)


def _balance_need_score(
    card: Card,
    props: CardProperties,
    sig: DeckSignature,
    deck: list[Card],
    floor: int,
) -> float:
    """Bonus for filling critical gaps in the deck.

    Kicks in earlier than before (floor 4 instead of 6), fires on a
    deeper definition of "block gap" that excludes starter Defends (since
    5 Defends aren't enough to survive Act 1 bosses), and carries bigger
    magnitudes so it can meaningfully sway picks.

    The 306-boss-fight data showed turn-4 and turn-8/10/11 as the top
    loss spikes — exactly the pattern you'd expect from decks arriving
    at the boss with no real block beyond starter Defends.
    """
    if floor < 4 or sig.picked_count < 2:
        return 0.0

    # Scale with how late we are; 0 at floor 3, 1.0 at floor 13
    late_factor = min(1.0, (floor - 3) / 10.0)

    bonus = 0.0

    # Block gap: count only NON-starter block cards. Starter Defends
    # always exist, so the original sig.block_card_count check never
    # fired for Silent decks.
    non_starter_block = sum(
        1 for c in deck
        if c.name.lower() not in _STARTER_NAMES
        and extract_properties(c).grants_block > 0
    )
    if non_starter_block == 0 and props.grants_block >= 8:
        bonus += 0.25 * late_factor

    # Draw gap: important for any deck that wants to see its good cards
    if sig.draw_card_count <= 1 and props.draws_cards > 0:
        bonus += 0.20 * late_factor

    # AoE gap: still important for multi-enemy fights
    if sig.aoe_count == 0 and props.is_aoe:
        bonus += 0.08 * late_factor

    return bonus


def _deck_size_penalty(sig: DeckSignature) -> float:
    """Penalty for adding cards to an already large deck.

    Every card added dilutes draw probability, but STS decks win by
    accumulating win conditions, so the early/mid run should feel no
    pressure at all.  The curve only starts to bite at 18+ cards and
    only gets punishing at 22+ (rare in Act 1, possible mid-Act 2).

    Previous curve (too aggressive):
      ≤15: 0  | 16–17: -0.05 | 18–19: -0.12 | 20+: -0.22
    New curve (deck growth encouraged):
      ≤17: 0  | 18–19: -0.04 | 20–21: -0.10 | 22–23: -0.16 | 24+: -0.22
    """
    size = sig.size
    if size <= 17:
        return 0.0
    elif size <= 19:
        return 0.04
    elif size <= 21:
        return 0.10
    elif size <= 23:
        return 0.16
    else:
        return 0.22


#: Power threshold above which a non-Power card is considered "strong"
#: and exempt from duplicate diversity pressure.  Calibrated so Backflip,
#: Bouncing Flask, Dagger Spray, Leg Sweep, Catalyst, and similar
#: archetype payoffs clear the bar, while starter-tier filler doesn't.
_STRONG_CARD_THRESHOLD: float = 0.35


def _duplicate_penalty(
    card: Card,
    deck: list[Card],
    props: CardProperties,
    power_score: float,
) -> float:
    """Penalty for having too many copies of the same card.

    Design:
      * Powers keep the full penalty — stacking duplicate Power effects
        rarely adds value (Noxious Fumes 2x, Accuracy 2x don't combine).
      * Strong non-Power cards: **zero penalty**.  Multiples of a good
        archetype payoff are how you build winning STS decks.  The card
        still has to earn its slot on its own power + alignment score,
        but duplication is no longer penalised at all.
      * Weak / mid cards: smoothly scaling diversity pressure — a pure
        Strike duplicate still eats most of the penalty, a mid card
        gets reduced pressure.
    """
    copies = sum(1 for c in deck if c.id == card.id or c.name == card.name)
    if copies == 0:
        return 0.0

    # Powers: stacking duplicate Powers rarely adds value.
    if props.is_power:
        return 0.25 if copies >= 2 else 0.08

    # Strong non-Power cards: no penalty at all.
    if power_score >= _STRONG_CARD_THRESHOLD:
        return 0.0

    # Weak / mid cards: smooth diversity pressure
    base = 0.25 if copies >= 2 else 0.08
    scale = max(0.10, 1.0 - power_score)
    return base * scale


def score_card(
    card: Card,
    deck: list[Card],
    floor: int,
    hp: int = 50,
    max_hp: int = 70,
    relics: frozenset[str] | set[str] | None = None,
) -> float:
    """Score a card for the pick decision. Higher = better to pick.

    Combines:
      - Intrinsic card power (good in any deck)
      - Alignment with deck direction (synergy / off-archetype penalty)
      - Balance gap filling (late-game only)
      - Deck size penalty (dilution cost)
      - Duplicate penalty (non-Power strong cards are exempt)
      - Relic synergy bonus (Shuriken likes cheap attacks, Paper Krane
        likes Weak, Snecko Skull likes poison, etc.) when ``relics`` is
        supplied
    """
    props = extract_properties(card)
    sig = build_signature(deck)

    power = _card_power_score(card, props)
    alignment = _alignment_score(card, props, sig)
    balance = _balance_need_score(card, props, sig, deck, floor)
    size_pen = _deck_size_penalty(sig)
    dup_pen = _duplicate_penalty(card, deck, props, power)

    # HP factor: when low on HP, value defensive cards more.
    # Triggers at 60% HP (was 40%) and the bonus is doubled — the
    # boss-fight data showed turn-4 deaths as the #1 loss mode, which
    # means decks are arriving at bosses without enough block.
    hp_ratio = hp / max(1, max_hp)
    hp_defense_bonus = 0.0
    if hp_ratio < 0.6 and props.grants_block > 0:
        hp_defense_bonus = 0.10 * (1.0 - hp_ratio)

    # Relic bonus: amplify cards whose mechanical effects match owned
    # relics.  Caps at ±0.25 inside relic_card_bonus so it nudges rather
    # than dominates.  Imported lazily to avoid a circular import during
    # module load.
    relic_bonus = 0.0
    if relics:
        from .relic_synergy import relic_card_bonus
        relic_bonus = relic_card_bonus(props, relics)

    # V7: Act-1 premium-neutral bonus.  Empirically-derived from the V6
    # boss-log: these cards have the highest win-lift in Act 1 and are
    # all neutral generalists.  Small +0.10 nudge, Act 1 only.
    # FIX 1: floor boundary changed from 17 to 15
    premium_bonus = 0.0
    if floor < 15 and _is_premium_neutral(card):
        premium_bonus = 0.10

    score = (
        power + alignment + balance + hp_defense_bonus + relic_bonus
        + premium_bonus - size_pen - dup_pen
    )
    return max(0.0, min(1.0, score))


def score_skip(deck: list[Card], floor: int) -> float:
    """Score for skipping the card reward.

    V7 curve (retuned against V6 boss-log data showing winning decks
    average 16.8 cards vs losing 16.2): stay hungry below 17, ramp up
    fast above 18.  The previous curve was too flat at 18-20, letting
    decks bloat to 19-20 and then dilute the starter core.

    Also aligned with _deck_size_penalty so they both start biting in
    the same range.
    """
    sig = build_signature(deck)
    size = sig.size

    # V7: steeper ramp above 18, unchanged below 17
    if size >= 24:
        base = 0.50
    elif size >= 22:
        base = 0.42
    elif size >= 20:
        base = 0.33
    elif size >= 19:
        base = 0.24
    elif size >= 18:
        base = 0.17
    elif size >= 17:
        base = 0.10
    elif size >= 16:
        base = 0.05
    else:
        base = 0.02  # Almost never skip below 16 cards

    # V7: commitment boost dropped 0.08 -> 0.04 because the new
    # commitment threshold is much higher and firing this at full
    # commitment (5+ archetype cards) would push skip too aggressively
    # right when the deck is finally working.
    base += sig.archetype_commitment * 0.04

    return min(0.55, base)


# ---------------------------------------------------------------------------
# Public pick interface
# ---------------------------------------------------------------------------

def pick_card(
    offered: list[Card],
    deck: list[Card],
    floor: int = 1,
    hp: int = 50,
    max_hp: int = 70,
    relics: frozenset[str] | set[str] | None = None,
) -> Card | None:
    """Pick the best card from offered rewards, or None to skip.

    Pure rule-based organic scoring: score_card() for each offered card
    versus score_skip() as the baseline.  When ``relics`` is supplied,
    the relic synergy bonus is applied inside score_card — relic-aware
    callers should pass the player's current relic set so that, e.g.,
    a Paper Krane deck preferentially takes Weak-applying cards.
    """
    if not offered:
        return None

    # Score all options + skip
    scores: list[tuple[Card | None, float]] = [
        (card, score_card(card, deck, floor, hp, max_hp, relics=relics))
        for card in offered
    ]
    scores.append((None, score_skip(deck, floor)))

    # Pick the highest
    scores.sort(key=lambda x: x[1], reverse=True)
    best_card, _ = scores[0]

    return best_card  # None = skip
