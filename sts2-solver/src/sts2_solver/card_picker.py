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
  - Alpha-blended interface: rule score * (1-alpha) + ml score * alpha
    where alpha ramps up as wins accumulate

The ML layer (XGBoost residual) is pluggable and starts at zero weight.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
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
        """The emergent archetype, or 'undecided' if none dominates."""
        scores = {
            "poison": self.poison_card_count,
            "shiv": self.shiv_card_count,
            "sly": self.sly_card_count,
        }
        best = max(scores, key=scores.get)
        best_count = scores[best]
        # Need at least 2 cards for a direction to emerge
        if best_count < 2:
            return "undecided"
        # Need some separation from second-best
        second = sorted(scores.values(), reverse=True)[1]
        if best_count <= second:
            return "mixed"
        return best

    @property
    def archetype_commitment(self) -> float:
        """0.0 = no direction, 1.0 = fully committed."""
        scores = [self.poison_card_count, self.shiv_card_count, self.sly_card_count]
        best = max(scores)
        if best == 0 or self.picked_count == 0:
            return 0.0
        return min(1.0, best / max(1, self.picked_count))

    @property
    def defense_ratio(self) -> float:
        """Fraction of deck that provides block."""
        return self.block_card_count / max(1, self.size)

    @property
    def draw_ratio(self) -> float:
        """Fraction of deck that draws cards."""
        return self.draw_card_count / max(1, self.size)


_STARTER_NAMES = frozenset({"strike", "defend", "survivor", "neutralize"})


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

    Returns -0.3 (off-archetype) to +0.4 (perfect fit).
    Magnitude scales with deck commitment — early on, everything is ~0.
    """
    commitment = sig.archetype_commitment
    if commitment < 0.1:
        return 0.0  # No direction yet — all cards equally valid

    archetype = sig.dominant_archetype
    if archetype == "undecided" or archetype == "mixed":
        return 0.0

    # Check if card fits the dominant archetype (by properties, not names)
    is_in_archetype = False
    if archetype == "poison" and props.applies_poison > 0:
        is_in_archetype = True
    elif archetype == "shiv" and props.spawns_shivs:
        is_in_archetype = True
    elif archetype == "sly" and props.has_sly:
        is_in_archetype = True

    # Cross-synergies: Sly supports both poison and shiv
    is_cross_synergy = False
    if archetype in ("poison", "shiv") and props.has_sly:
        is_cross_synergy = True
    # Shiv generators synergise with Sly (more cards to play)
    if archetype == "sly" and props.spawns_shivs:
        is_cross_synergy = True

    # Draw and energy cards support every archetype
    is_universal_support = props.draws_cards > 0 or props.grants_energy > 0

    if is_in_archetype:
        return 0.5 * commitment   # Max +0.5 at full commitment
    elif is_cross_synergy:
        return 0.25 * commitment  # Cross-synergy still valuable
    elif is_universal_support:
        return 0.1 * commitment   # Draw/energy always helps
    else:
        # Off-archetype penalty scales with commitment
        return -0.35 * commitment  # Max -0.35 at full commitment


def _balance_need_score(
    card: Card,
    props: CardProperties,
    sig: DeckSignature,
    floor: int,
) -> float:
    """Bonus for filling critical gaps in the deck.

    Per maxim #3, this is small early and grows as the deck matures.
    It should never override a strong synergy pick, but should break ties
    and prevent critical deficiencies.
    """
    # Only start caring about balance after floor 6 and a few picks
    if floor < 6 or sig.picked_count < 3:
        return 0.0

    # Scale with how late we are
    late_factor = min(1.0, (floor - 5) / 10.0)  # 0.0 at floor 5, 1.0 at floor 15

    bonus = 0.0

    # Critical: no block cards at all
    if sig.block_card_count <= 1 and props.grants_block > 0:
        bonus += 0.15 * late_factor

    # Important: no draw cards
    if sig.draw_card_count <= 1 and props.draws_cards > 0:
        bonus += 0.10 * late_factor

    # AoE needed for multi-enemy fights
    if sig.aoe_count == 0 and props.is_aoe:
        bonus += 0.08 * late_factor

    return bonus


def _deck_size_penalty(sig: DeckSignature) -> float:
    """Penalty for adding cards to an already large deck.

    Every card added dilutes draw probability. The nth card needs to be
    increasingly good to justify the slot. A normal Silent deck is 15-17
    cards by end of Act 1 — penalty should be gentle until 18+.
    """
    size = sig.size
    if size <= 15:
        return 0.0
    elif size <= 17:
        return 0.05
    elif size <= 19:
        return 0.12
    else:
        return 0.22  # Hard to justify adding to a 20+ card deck


def _duplicate_penalty(card: Card, deck: list[Card]) -> float:
    """Penalty for having too many copies of the same card."""
    copies = sum(1 for c in deck if c.id == card.id or c.name == card.name)
    if copies >= 2:
        return 0.25
    elif copies >= 1:
        return 0.08
    return 0.0


def score_card(
    card: Card,
    deck: list[Card],
    floor: int,
    hp: int = 50,
    max_hp: int = 80,
) -> float:
    """Score a card for the pick decision. Higher = better to pick.

    Combines:
      - Intrinsic card power (good in any deck)
      - Alignment with deck direction (synergy / off-archetype penalty)
      - Balance gap filling (late-game only)
      - Deck size penalty (dilution cost)
      - Duplicate penalty
    """
    props = extract_properties(card)
    sig = build_signature(deck)

    power = _card_power_score(card, props)
    alignment = _alignment_score(card, props, sig)
    balance = _balance_need_score(card, props, sig, floor)
    size_pen = _deck_size_penalty(sig)
    dup_pen = _duplicate_penalty(card, deck)

    # HP factor: when low on HP, value defensive cards more
    hp_ratio = hp / max(1, max_hp)
    hp_defense_bonus = 0.0
    if hp_ratio < 0.4 and props.grants_block > 0:
        hp_defense_bonus = 0.05 * (1.0 - hp_ratio)

    score = power + alignment + balance + hp_defense_bonus - size_pen - dup_pen
    return max(0.0, min(1.0, score))


def score_skip(deck: list[Card], floor: int) -> float:
    """Score for skipping the card reward.

    Skipping should be rare early (deck needs to grow from 10 starters to
    ~15-17 cards) and increasingly common once the deck has an identity.
    A 14-card deck is normal and healthy — skip pressure should only kick
    in hard above 17-18 cards.
    """
    sig = build_signature(deck)
    size = sig.size

    # Skipping is better when deck is large and focused
    base = 0.0
    if size >= 20:
        base = 0.45
    elif size >= 18:
        base = 0.35
    elif size >= 16:
        base = 0.25
    elif size >= 14:
        base = 0.12
    else:
        base = 0.05  # Almost never skip with < 14 cards

    # High commitment = skip off-archetype offers more readily
    base += sig.archetype_commitment * 0.08

    return min(0.55, base)


# ---------------------------------------------------------------------------
# Alpha-blended picker (rule + ML)
# ---------------------------------------------------------------------------

_ML_MODEL = None
_TOTAL_WINS: int = 0
_WIN_THRESHOLD: int = 500  # wins needed for full ML handoff


def set_win_count(wins: int) -> None:
    """Update the global win count for alpha calculation."""
    global _TOTAL_WINS
    _TOTAL_WINS = wins


def get_alpha() -> float:
    """Current blend weight: 0.0 = pure rules, 1.0 = pure ML."""
    return min(1.0, _TOTAL_WINS / _WIN_THRESHOLD)


def load_ml_model(path: str | Path | None = None) -> bool:
    """Load the XGBoost residual model if available."""
    global _ML_MODEL
    if path is None:
        path = Path(__file__).resolve().parents[3] / "card_picker_model" / "card_picker.json"
    path = Path(path)
    if not path.exists():
        return False
    try:
        from .card_picker_xgb import CardPickerXGB
        _ML_MODEL = CardPickerXGB(path)
        return True
    except Exception:
        return False


def _ml_score(
    card: Card | None,
    deck: list[Card],
    floor: int,
    hp: int,
    max_hp: int,
) -> float:
    """Get ML model score for a card (or skip if card is None)."""
    if _ML_MODEL is None:
        return 0.0
    try:
        from .card_picker_xgb import build_feature_row, build_skip_features, feats_to_array
        import numpy as np
        if card is not None:
            feats = build_feature_row(card, deck, floor, hp, max_hp)
        else:
            feats = build_skip_features(deck, floor, hp, max_hp)
        x = feats_to_array(feats).reshape(1, -1)
        return float(_ML_MODEL.model.predict(x)[0])
    except Exception:
        return 0.0


def blended_score(
    card: Card | None,
    deck: list[Card],
    floor: int,
    hp: int = 50,
    max_hp: int = 80,
) -> float:
    """Score a card using alpha-blended rules + ML.

    card=None scores the skip option.
    """
    alpha = get_alpha()

    if card is not None:
        rule = score_card(card, deck, floor, hp, max_hp)
    else:
        rule = score_skip(deck, floor)

    if alpha == 0.0 or _ML_MODEL is None:
        return rule

    ml = _ml_score(card, deck, floor, hp, max_hp)
    return (1.0 - alpha) * rule + alpha * ml


# ---------------------------------------------------------------------------
# Public pick interface
# ---------------------------------------------------------------------------

def pick_card(
    offered: list[Card],
    deck: list[Card],
    floor: int = 1,
    hp: int = 50,
    max_hp: int = 80,
) -> Card | None:
    """Pick the best card from offered rewards, or None to skip.

    This is the main entry point — drop-in replacement for
    simulator._pick_card_reward and the XGBoost picker.
    """
    if not offered:
        return None

    # Score all options + skip
    scores = []
    for card in offered:
        s = blended_score(card, deck, floor, hp, max_hp)
        scores.append((card, s))

    skip_score = blended_score(None, deck, floor, hp, max_hp)
    scores.append((None, skip_score))

    # Pick the highest
    scores.sort(key=lambda x: x[1], reverse=True)
    best_card, best_score = scores[0]

    return best_card  # None = skip
