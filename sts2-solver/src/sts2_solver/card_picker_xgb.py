"""XGBoost-based card reward picker for Slay the Spire 2 (Silent).

Replaces the hardcoded tier-list picker with a model that learns which
cards lead to wins based on deck composition, floor, HP, and card features.

Pipeline:
    1. collect_data()  — run N self-play games, log every card-pick decision
    2. train()         — train an XGBoost ranker on the logged data
    3. pick()          — at runtime, score offered cards and pick the best

The model is a pointwise regressor: for each (deck_context, offered_card) pair
it predicts the probability of winning the run if that card is picked.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .models import Card
from .constants import CardType, TargetType
from .card_picker import extract_properties as _extract_props


# ---------------------------------------------------------------------------
# Property-based archetype detection helpers
# ---------------------------------------------------------------------------

def _is_poison_card(card: Card) -> bool:
    """Card applies or interacts with poison (detected from card data)."""
    return _extract_props(card).applies_poison > 0

def _is_shiv_card(card: Card) -> bool:
    """Card spawns shivs."""
    return _extract_props(card).spawns_shivs

def _is_sly_card(card: Card) -> bool:
    """Card has Sly keyword."""
    return _extract_props(card).has_sly

def _is_draw_card(card: Card) -> bool:
    """Card draws cards."""
    return _extract_props(card).draws_cards > 0

def _is_defense_card(card: Card) -> bool:
    """Card provides block or dexterity."""
    props = _extract_props(card)
    return props.grants_block > 0 or props.grants_dexterity


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _card_type_int(ct: CardType) -> int:
    mapping = {CardType.ATTACK: 0, CardType.SKILL: 1, CardType.POWER: 2}
    return mapping.get(ct, 3)


def _deck_features(deck: list[Card], floor: int, hp: int, max_hp: int) -> dict[str, float]:
    """Extract deck-composition features for the current game state."""
    deck_size = len(deck)

    # Card type counts
    n_attacks = sum(1 for c in deck if c.card_type == CardType.ATTACK)
    n_skills = sum(1 for c in deck if c.card_type == CardType.SKILL)
    n_powers = sum(1 for c in deck if c.card_type == CardType.POWER)

    # Archetype counts — property-based, not name-based
    n_poison = sum(1 for c in deck if _is_poison_card(c))
    n_shiv = sum(1 for c in deck if _is_shiv_card(c))
    n_sly = sum(1 for c in deck if _is_sly_card(c))
    n_draw = sum(1 for c in deck if _is_draw_card(c))
    n_defense = sum(1 for c in deck if _is_defense_card(c))

    # Cost distribution
    costs = [c.cost for c in deck if c.cost >= 0]
    avg_cost = np.mean(costs) if costs else 1.0
    zero_cost = sum(1 for c in costs if c == 0)

    # Damage / block totals
    total_damage = sum(c.damage or 0 for c in deck)
    total_block = sum(c.block or 0 for c in deck)
    total_draw = sum(c.cards_draw for c in deck)

    # Upgraded fraction
    n_upgraded = sum(1 for c in deck if c.upgraded)

    return {
        "floor": floor,
        "hp_frac": hp / max(1, max_hp),
        "deck_size": deck_size,
        "n_attacks": n_attacks,
        "n_skills": n_skills,
        "n_powers": n_powers,
        "attack_frac": n_attacks / max(1, deck_size),
        "skill_frac": n_skills / max(1, deck_size),
        "power_frac": n_powers / max(1, deck_size),
        "n_poison": n_poison,
        "n_shiv": n_shiv,
        "n_sly": n_sly,
        "n_draw": n_draw,
        "n_defense": n_defense,
        "poison_frac": n_poison / max(1, deck_size),
        "shiv_frac": n_shiv / max(1, deck_size),
        "sly_frac": n_sly / max(1, deck_size),
        "avg_cost": avg_cost,
        "zero_cost_cards": zero_cost,
        "total_damage": total_damage,
        "total_block": total_block,
        "total_draw": total_draw,
        "n_upgraded": n_upgraded,
        "upgraded_frac": n_upgraded / max(1, deck_size),
    }


def _card_features(card: Card, deck: list[Card]) -> dict[str, float]:
    """Extract features for a single candidate card."""
    card_name = card.name.lower()

    # How many copies already in deck
    copies_in_deck = sum(1 for c in deck if c.name.lower() == card_name)

    # Archetype membership — property-based
    is_poison = 1.0 if _is_poison_card(card) else 0.0
    is_shiv = 1.0 if _is_shiv_card(card) else 0.0
    is_sly = 1.0 if _is_sly_card(card) else 0.0
    is_draw = 1.0 if _is_draw_card(card) else 0.0
    is_defense = 1.0 if _is_defense_card(card) else 0.0

    # Card stats
    damage = card.damage or 0
    block = card.block or 0
    cost = card.cost if card.cost >= 0 else 0
    draw = card.cards_draw
    energy_gain = card.energy_gain
    hit_count = card.hit_count
    exhausts = 1.0 if card.exhausts else 0.0
    is_power = 1.0 if card.card_type == CardType.POWER else 0.0
    is_aoe = 1.0 if card.target == TargetType.ALL_ENEMIES else 0.0
    is_upgraded = 1.0 if card.upgraded else 0.0

    # Synergy scores: how well does this card fit the existing deck
    n_poison = sum(1 for c in deck if _is_poison_card(c))
    n_shiv = sum(1 for c in deck if _is_shiv_card(c))
    n_sly = sum(1 for c in deck if _is_sly_card(c))

    poison_synergy = is_poison * n_poison
    shiv_synergy = is_shiv * n_shiv
    sly_synergy = is_sly * n_sly

    # Poison amount from card properties (covers vars + powers_applied)
    props = _extract_props(card)
    applies_poison = props.applies_poison

    return {
        "card_cost": cost,
        "card_damage": damage,
        "card_block": block,
        "card_draw": draw,
        "card_energy_gain": energy_gain,
        "card_hit_count": hit_count,
        "card_exhausts": exhausts,
        "card_is_power": is_power,
        "card_is_aoe": is_aoe,
        "card_is_upgraded": is_upgraded,
        "card_is_poison": is_poison,
        "card_is_shiv": is_shiv,
        "card_is_sly": is_sly,
        "card_is_draw": is_draw,
        "card_is_defense": is_defense,
        "card_copies_in_deck": copies_in_deck,
        "card_poison_synergy": poison_synergy,
        "card_shiv_synergy": shiv_synergy,
        "card_sly_synergy": sly_synergy,
        "card_applies_poison": applies_poison,
        "card_type_int": _card_type_int(card.card_type),
    }


def build_feature_row(
    card: Card,
    deck: list[Card],
    floor: int,
    hp: int,
    max_hp: int,
) -> dict[str, float]:
    """Build a full feature dict for one (deck_context, candidate_card) pair."""
    feats = _deck_features(deck, floor, hp, max_hp)
    feats.update(_card_features(card, deck))
    return feats


# Canonical feature order (for consistent numpy arrays)
FEATURE_NAMES: list[str] = list(build_feature_row(
    Card(id="dummy", name="Dummy", cost=1, card_type=CardType.ATTACK,
         target=TargetType.ANY_ENEMY),
    [], 1, 50, 80,
).keys())


def feats_to_array(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[k] for k in FEATURE_NAMES], dtype=np.float32)


def build_skip_features(
    deck: list[Card], floor: int, hp: int, max_hp: int,
) -> dict[str, float]:
    """Feature row for the 'skip' option (no card picked)."""
    feats = _deck_features(deck, floor, hp, max_hp)
    # Card features are all zero for skip
    for k in FEATURE_NAMES:
        if k.startswith("card_"):
            feats.setdefault(k, 0.0)
    return feats


# ---------------------------------------------------------------------------
# Data collection: log card picks during self-play
# ---------------------------------------------------------------------------

@dataclass
class CardPickRecord:
    """One card-pick decision point."""
    floor: int
    hp: int
    max_hp: int
    deck_names: list[str]
    offered_names: list[str]
    picked_name: str | None   # None = skip
    run_outcome: str = ""     # "win" or "lose" — filled post-run
    floor_reached: int = 0    # filled post-run


@dataclass
class CardPickCollector:
    """Accumulates card pick records across a training run."""
    records: list[CardPickRecord] = field(default_factory=list)

    def log_pick(
        self,
        floor: int,
        hp: int,
        max_hp: int,
        deck: list[Card],
        offered: list[Card],
        picked: Card | None,
    ) -> None:
        self.records.append(CardPickRecord(
            floor=floor,
            hp=hp,
            max_hp=max_hp,
            deck_names=[c.name for c in deck],
            offered_names=[c.name for c in offered],
            picked_name=picked.name if picked else None,
        ))

    def finalize_run(self, outcome: str, floor_reached: int) -> None:
        """Tag all records from this run with the outcome."""
        for r in self.records:
            if not r.run_outcome:
                r.run_outcome = outcome
                r.floor_reached = floor_reached

    def to_dicts(self) -> list[dict]:
        return [
            {
                "floor": r.floor,
                "hp": r.hp,
                "max_hp": r.max_hp,
                "deck": r.deck_names,
                "offered": r.offered_names,
                "picked": r.picked_name,
                "outcome": r.run_outcome,
                "floor_reached": r.floor_reached,
            }
            for r in self.records
        ]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(self.to_dicts(), f)
        print(f"Saved {len(self.records)} card pick records to {p}")


# ---------------------------------------------------------------------------
# Training data conversion
# ---------------------------------------------------------------------------

def _synergy_score(card: Card, deck: list[Card]) -> float:
    """Score how well a card fits the existing deck's archetype.

    Returns 0.0–1.0 where higher means stronger synergy fit.
    This provides training signal even without wins — the model learns
    that poison cards belong in poison decks, shivs with shivs, etc.

    All detection is property-based (via extract_properties), not name lists.
    """
    deck_size = len(deck)
    if deck_size == 0:
        return 0.3  # neutral for empty deck

    # Count existing archetype cards by properties
    n_poison = sum(1 for c in deck if _is_poison_card(c))
    n_shiv = sum(1 for c in deck if _is_shiv_card(c))
    n_sly = sum(1 for c in deck if _is_sly_card(c))

    # Detect dominant archetype (if any)
    archetype_counts = {"poison": n_poison, "shiv": n_shiv, "sly": n_sly}
    dominant = max(archetype_counts, key=archetype_counts.get)
    dominant_count = archetype_counts[dominant]

    # Card's archetype membership by properties
    card_is_poison = _is_poison_card(card)
    card_is_shiv = _is_shiv_card(card)
    card_is_sly = _is_sly_card(card)
    card_is_draw = _is_draw_card(card)
    card_is_defense = _is_defense_card(card)

    # No clear archetype yet (< 2 cards) → slight bonus for any archetype card
    if dominant_count < 2:
        if card_is_poison or card_is_shiv or card_is_sly:
            return 0.35
        if card_is_draw:
            return 0.3  # draw is always decent
        return 0.2

    # Has an emerging archetype → reward in-archetype, penalize off-archetype
    is_in_archetype = (
        (dominant == "poison" and card_is_poison) or
        (dominant == "shiv" and card_is_shiv) or
        (dominant == "sly" and card_is_sly)
    )

    # Cross-archetype synergies (Sly supports both poison and shiv)
    is_cross_synergy = (dominant in ("poison", "shiv") and card_is_sly)

    if is_in_archetype:
        # Stronger bonus the more committed the deck is
        commitment = min(1.0, dominant_count / 5.0)
        return 0.5 + 0.3 * commitment  # 0.5–0.8
    elif is_cross_synergy:
        return 0.45
    elif card_is_draw:
        return 0.35
    elif card_is_defense:
        return 0.3
    else:
        # Off-archetype card — penalize more as deck commitment grows
        commitment = min(1.0, dominant_count / 5.0)
        return max(0.05, 0.25 - 0.15 * commitment)  # 0.25–0.10


def _skip_score(deck: list[Card], deck_size_target: int = 12) -> float:
    """Score for skipping a card reward. Higher = skipping is better.

    Skipping is valuable when the deck is already large or well-focused.
    """
    deck_size = len(deck)
    if deck_size >= deck_size_target + 3:
        return 0.6  # strongly prefer skip when bloated
    elif deck_size >= deck_size_target:
        return 0.4  # lean toward skip
    elif deck_size <= 8:
        return 0.1  # almost always pick when deck is tiny
    return 0.25


def records_to_training_data(
    records_path: str | Path,
    card_db: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert logged records to (X, y) training arrays.

    Labels combine outcome signal with synergy scoring:
      - Every offered card gets a synergy score (0.0–0.8) based on
        how well it fits the deck's emerging archetype
      - The picked card gets an additional outcome bonus from floor progress
      - This gives the model signal about archetype fit even without wins

    Label formula:
      base = synergy_score(card, deck)       # 0.0–0.8 for all cards
      if was_picked: base += outcome_bonus   # extra credit from run result
      label = clamp(base, 0.0, 1.0)
    """
    with open(records_path) as f:
        records = json.load(f)

    rows_X = []
    rows_y = []

    for rec in records:
        floor = rec["floor"]
        hp = rec["hp"]
        max_hp = rec["max_hp"]
        outcome = rec["outcome"]
        floor_reached = rec["floor_reached"]
        picked_name = rec["picked"]

        # Reconstruct deck as Card objects
        deck = _names_to_cards(rec["deck"], card_db)
        offered = _names_to_cards(rec["offered"], card_db)

        if not offered:
            continue

        # Outcome bonus for the picked card
        if outcome == "win":
            outcome_bonus = 0.3
        else:
            outcome_bonus = 0.15 * (floor_reached / 17.0)

        # One row per offered card — ALL cards get synergy scores
        for card in offered:
            feats = build_feature_row(card, deck, floor, hp, max_hp)
            synergy = _synergy_score(card, deck)
            was_picked = (card.name == picked_name)
            label = synergy + (outcome_bonus if was_picked else 0.0)
            label = max(0.0, min(1.0, label))
            rows_X.append(feats_to_array(feats))
            rows_y.append(label)

        # Skip row
        skip_feats = build_skip_features(deck, floor, hp, max_hp)
        skip_syn = _skip_score(deck)
        was_skip = (picked_name is None)
        skip_label = skip_syn + (outcome_bonus if was_skip else 0.0)
        skip_label = max(0.0, min(1.0, skip_label))
        rows_X.append(feats_to_array(skip_feats))
        rows_y.append(skip_label)

    X = np.vstack(rows_X) if rows_X else np.zeros((0, len(FEATURE_NAMES)))
    y = np.array(rows_y, dtype=np.float32)
    return X, y


def _names_to_cards(names: list[str], card_db: Any) -> list[Card]:
    """Look up Card objects by name from card_db."""
    # Build a name→Card lookup (cached on the card_db instance)
    if not hasattr(card_db, '_name_index'):
        index: dict[str, Card] = {}
        all_cards = (card_db.all_cards() if hasattr(card_db, 'all_cards')
                     else card_db.values() if hasattr(card_db, 'values')
                     else [])
        for c in all_cards:
            key = c.name.lower()
            if key not in index:  # prefer base over upgraded
                index[key] = c
        card_db._name_index = index

    cards = []
    for name in names:
        found = card_db._name_index.get(name.lower())
        if found:
            cards.append(found)
    return cards


# ---------------------------------------------------------------------------
# XGBoost model wrapper
# ---------------------------------------------------------------------------

MODEL_DIR = Path(__file__).resolve().parents[3] / "card_picker_model"


class CardPickerXGB:
    """Wraps an XGBoost regressor for card pick decisions."""

    def __init__(self, model_path: str | Path | None = None):
        self.model = None
        if model_path:
            self.load(model_path)

    def train(self, X: np.ndarray, y: np.ndarray, save_path: str | Path | None = None):
        """Train the XGBoost model on collected data."""
        import xgboost as xgb

        self.model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=5,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=42,
        )

        # 80/20 train/eval split
        n = len(X)
        idx = np.random.RandomState(42).permutation(n)
        split = int(0.8 * n)
        X_train, X_eval = X[idx[:split]], X[idx[split:]]
        y_train, y_eval = y[idx[:split]], y[idx[split:]]

        self.model.fit(
            X_train, y_train,
            eval_set=[(X_eval, y_eval)],
            verbose=True,
        )

        if save_path:
            p = Path(save_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(str(p))
            # Also save feature names for reference
            meta_path = p.with_suffix(".meta.json")
            with open(meta_path, "w") as f:
                json.dump({"feature_names": FEATURE_NAMES, "n_samples": n}, f)
            print(f"Model saved to {p} ({n} samples)")

    def load(self, path: str | Path):
        import xgboost as xgb
        self.model = xgb.XGBRegressor()
        self.model.load_model(str(path))

    def score_cards(
        self,
        offered: list[Card],
        deck: list[Card],
        floor: int,
        hp: int,
        max_hp: int,
    ) -> list[tuple[Card | None, float]]:
        """Score each offered card + skip. Returns sorted (card, score) list."""
        if self.model is None:
            raise RuntimeError("Model not loaded — call train() or load() first")

        results = []
        for card in offered:
            feats = build_feature_row(card, deck, floor, hp, max_hp)
            x = feats_to_array(feats).reshape(1, -1)
            score = float(self.model.predict(x)[0])
            results.append((card, score))

        # Skip option
        skip_feats = build_skip_features(deck, floor, hp, max_hp)
        x_skip = feats_to_array(skip_feats).reshape(1, -1)
        skip_score = float(self.model.predict(x_skip)[0])
        results.append((None, skip_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def pick(
        self,
        offered: list[Card],
        deck: list[Card],
        floor: int,
        hp: int,
        max_hp: int,
    ) -> Card | None:
        """Pick the best card (or None to skip)."""
        if not offered:
            return None
        scored = self.score_cards(offered, deck, floor, hp, max_hp)
        best_card, best_score = scored[0]
        return best_card  # None means skip


# ---------------------------------------------------------------------------
# Global singleton for use during self-play / simulation
# ---------------------------------------------------------------------------

_GLOBAL_PICKER: CardPickerXGB | None = None


def get_picker() -> CardPickerXGB | None:
    """Get the global XGBoost picker, loading it if available."""
    global _GLOBAL_PICKER
    if _GLOBAL_PICKER is not None:
        return _GLOBAL_PICKER

    model_path = MODEL_DIR / "card_picker.json"
    if model_path.exists():
        try:
            _GLOBAL_PICKER = CardPickerXGB(model_path)
            print(f"[CardPicker] Loaded XGBoost model from {model_path}")
            return _GLOBAL_PICKER
        except Exception as e:
            print(f"[CardPicker] Failed to load model: {e}")
    return None


def xgb_pick_card_reward(
    offered: list[Card],
    deck: list[Card],
    floor: int = 1,
    hp: int = 50,
    max_hp: int = 70,
) -> Card | None:
    """Drop-in replacement for simulator._pick_card_reward.

    Falls back to the original tier-list picker if XGBoost model isn't available.
    """
    picker = get_picker()
    if picker is None:
        # Fallback: import and use the original
        from .simulator import _pick_card_reward
        return _pick_card_reward(offered, deck)

    return picker.pick(offered, deck, floor, hp, max_hp)
