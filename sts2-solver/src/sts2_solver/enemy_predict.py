"""Enemy intent prediction using move table lookahead.

Matches an enemy's current observed intent against its move table to
infer where it is in its cycle, then predicts the next N intents.
When the runner tracks move indices across turns, the known index is
used directly — no guessing needed.
"""

from __future__ import annotations

from .models import EnemyState
from .simulator import ENEMY_MOVE_TABLES


def _match_move_index(enemy_id: str, intent_type: str | None,
                      intent_damage: int | None, intent_hits: int) -> int | None:
    """Find the most likely current move index by matching observed intent.

    Returns the index into the move table that best matches the current
    intent, or None if no match is found.

    Fix 8: Includes fuzzy matching with tolerance for damage variations
    due to Strength buffs or other enemy modifications.
    """
    table = ENEMY_MOVE_TABLES.get(enemy_id)
    if not table or intent_type is None:
        return None

    # Normalize intent types: the game sometimes sends variants like
    # "DebuffStrong" that should match our table's "Debuff", or
    # "Sleep"/"Summon" which map to "Buff"/"Debuff" in our tables.
    # "StatusCard" is a pre-combat indicator that can mean any non-attack.
    _INTENT_NORMALIZE = {
        "DebuffStrong": "Debuff",
        "Sleep": "Buff",       # passive/setup intent
        "Summon": "Buff",      # summoning is a non-attack action
        "StatusCard": "Debuff",  # status card intent → debuff category
    }
    intent_type = _INTENT_NORMALIZE.get(intent_type, intent_type)

    best_idx = None
    best_score = -1
    fuzzy_match_used = False

    # First pass: exact matching
    for i, move in enumerate(table):
        score = 0
        # Type must match
        if move.get("type") != intent_type:
            continue
        score += 1

        # For attacks, match damage and hits
        if intent_type == "Attack":
            if intent_damage is not None and move.get("damage") == intent_damage:
                score += 2
            if intent_hits == move.get("hits", 1):
                score += 1

        if score > best_score:
            best_score = score
            best_idx = i

    # If exact match found, return it
    if best_idx is not None:
        return best_idx

    # Second pass: fuzzy matching with tolerance (±3 damage)
    # This handles cases where enemies have Strength buffs or other modifications
    damage_tolerance = 3
    for i, move in enumerate(table):
        score = 0
        # Type must match
        if move.get("type") != intent_type:
            continue
        score += 1

        # For attacks, fuzzy match damage and hits
        if intent_type == "Attack":
            if intent_damage is not None:
                move_damage = move.get("damage", 0)
                if abs(intent_damage - move_damage) <= damage_tolerance:
                    score += 2
                    # Give extra credit for closer matches
                    score += (damage_tolerance - abs(intent_damage - move_damage)) * 0.5
            if intent_hits == move.get("hits", 1):
                score += 1

        if score > best_score:
            best_score = score
            best_idx = i
            fuzzy_match_used = True

    # If fuzzy match found, log it for debugging
    if best_idx is not None and fuzzy_match_used:
        import sys
        move = table[best_idx]
        print(
            f"[enemy_predict] Fuzzy match for {enemy_id}: "
            f"observed damage={intent_damage}, matched move damage={move.get('damage')}",
            file=sys.stderr
        )

    # If still no match, fall back to the first move (default).
    # Only warn once per enemy ID to avoid log spam.
    if best_idx is None and table:
        best_idx = 0
        if not hasattr(_match_move_index, "_warned"):
            _match_move_index._warned = set()
        if enemy_id not in _match_move_index._warned:
            _match_move_index._warned.add(enemy_id)
            import sys
            print(
                f"[enemy_predict] No fuzzy match for {enemy_id} "
                f"(intent={intent_type}); using default move (index 0)",
                file=sys.stderr
        )

    return best_idx


def _synthesize_predictions(enemy: EnemyState, turns: int) -> list[dict]:
    """Synthesize future intent predictions from the enemy's current observed intent.

    Used when the enemy has no move table (e.g. Act 2 enemies we haven't
    catalogued yet).  Anchors entirely on what we can *see* this turn:

      - If current intent is Attack(X, hits):
            Use X as the baseline damage. Assume alternating: setup → attack.
            Scale X up slightly (+2) for future turns to account for
            strength gains we can't observe.
      - If current intent is Buff/Debuff/Defend/StatusCard:
            Estimate attack damage from enemy max_hp (higher HP → harder enemy).
            Alternate: attack → current_type → attack → ...
      - If intent is completely unknown:
            Scale damage estimate from enemy max_hp.

    The key insight: never use a hardcoded constant. The game is *telling*
    us how hard this enemy hits — listen to it.
    """
    result: list[dict] = []

    def _estimate_damage_from_hp(enemy: EnemyState) -> int:
        """Estimate likely attack damage from enemy max HP.

        Weak enemies (30-50 HP) hit for ~8-12.
        Normal enemies (50-80 HP) hit for ~12-18.
        Elite enemies (80-120 HP) hit for ~15-25.
        Boss enemies (120+ HP) hit for ~18-30.
        """
        max_hp = getattr(enemy, "max_hp", 0) or getattr(enemy, "hp", 50)
        if max_hp <= 40:
            return 10
        elif max_hp <= 70:
            return 15
        elif max_hp <= 120:
            return 20
        else:
            return 25

    if enemy.intent_type == "Attack" and enemy.intent_damage is not None:
        # Enemy is attacking now — use its ACTUAL damage as our baseline
        observed_dmg = enemy.intent_damage
        observed_hits = enemy.intent_hits or 1
        # Future attacks may be slightly stronger (strength gains)
        escalated_dmg = observed_dmg + 2

        attack_now = {"type": "Attack", "damage": observed_dmg, "hits": observed_hits}
        attack_later = {"type": "Attack", "damage": escalated_dmg, "hits": observed_hits}
        setup_move = {"type": "Buff", "self_strength": 2}

        for i in range(turns):
            if i == 0:
                result.append(dict(setup_move))
            elif i == 1:
                result.append(dict(attack_now))
            else:
                result.append(dict(attack_later if i % 2 == 1 else setup_move))

    elif enemy.intent_type in ("Buff", "Debuff", "Defend", "StatusCard"):
        # Non-damaging intent — estimate attack damage from enemy HP
        est_dmg = _estimate_damage_from_hp(enemy)
        attack_move = {"type": "Attack", "damage": est_dmg, "hits": 1}
        current_move = {"type": enemy.intent_type}
        for i in range(turns):
            result.append(dict(attack_move if i % 2 == 0 else current_move))

    else:
        # Completely unknown — estimate from HP
        est_dmg = _estimate_damage_from_hp(enemy)
        for _ in range(turns):
            result.append({
                "type": "Attack",
                "damage": est_dmg,
                "hits": 1,
            })

    return result


def predict_next_intents(enemy: EnemyState, turns: int = 2,
                         known_idx: int | None = None) -> list[dict]:
    """Predict the next N intents for an enemy based on its move table.

    If known_idx is provided (from runner tracking), uses it directly.
    Otherwise falls back to matching the current observed intent.

    For unknown enemies (no move table), synthesizes predictions from
    the enemy's current observed intent so MCTS/evaluator can still
    plan ahead instead of playing blind.
    """
    table = ENEMY_MOVE_TABLES.get(enemy.id)
    if not table:
        # No move table — synthesize from observed intent
        return _synthesize_predictions(enemy, turns)

    if known_idx is not None:
        idx = known_idx
    else:
        idx = _match_move_index(enemy.id, enemy.intent_type,
                                enemy.intent_damage, enemy.intent_hits)
        if idx is None:
            return []

    # Predict the next `turns` moves after the matched index
    result = []
    for offset in range(1, turns + 1):
        next_idx = (idx + offset) % len(table)
        result.append(dict(table[next_idx]))
    return result


def annotate_predictions(enemies: list[EnemyState], turns: int = 2,
                         move_indices: dict[tuple[int, str], int] | None = None) -> None:
    """Annotate a list of enemies with predicted future intents (in place).

    Args:
        enemies: List of enemies to annotate.
        turns: How many future intents to predict.
        move_indices: Optional dict of {(position, enemy_id): move_index}
            tracked by the runner across turns. When present, gives exact
            cycle position instead of guessing from intent matching.
    """
    for i, enemy in enumerate(enemies):
        if enemy.is_alive:
            known_idx = None
            if move_indices:
                known_idx = move_indices.get((i, enemy.id))
            enemy.predicted_intents = predict_next_intents(
                enemy, turns, known_idx=known_idx
            )
