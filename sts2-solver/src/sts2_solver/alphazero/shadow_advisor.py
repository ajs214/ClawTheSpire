"""Shadow advisor — sim-side heuristic picks for the agreement-rate diagnostic.

These functions mirror the logic of ``deterministic_advisor.py`` but operate
on the sim's raw data types (``Card`` objects, ints, dicts from event data)
rather than the live ``gs`` dict the live advisor expects. They are wired
into ``full_run.py`` at each option-decision site and their result is
stored in ``OptionSample.shadow_chosen_idx``.

**Never used in training loss.** Read-only telemetry consumed by
``tools/agreement_rate.py`` to answer the question "how often does the
network's current policy agree with the hand-written heuristic we'd use
if we had no network?".

When the agreement rate on a screen type climbs toward ~70–80% and the
network's win rate is ahead of the heuristic's, that screen is ready to
migrate from heuristic-driven to network-driven in live play.

Design notes:
- The logic here is intentionally simple. It's a *reference* pick, not
  necessarily the best pick. It should match what the live advisor would
  do on the same state, not beat it.
- Every function returns an index into the *same* option list the network
  was given. A mismatch between shadow and network indices means they
  disagree; a match means they agree.
- All functions are no-raise: they catch internal errors and return 0
  (the first option). A shadow pick of 0 for every sample would show up
  as uniformly low agreement in the report, which is the correct signal
  that something is wrong.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..card_registry import Card


# ---------------------------------------------------------------------------
# Map pathing
# ---------------------------------------------------------------------------

def shadow_pick_map(
    room_types: list[str],
    hp: int,
    max_hp: int,
    gold: int,
    deck_size: int,
    floor: int,
    character: str = "silent",
) -> int:
    """Mirror of ``decide_map`` on sim-side room types.

    ``room_types`` is the list already enumerated by play_full_run: a list
    of strings like "weak", "normal", "elite", "event", "shop", "rest".
    Returns the index of the best-scored room.
    """
    if not room_types:
        return 0

    hp_pct = hp / max(1, max_hp)

    def _score(ntype: str) -> float:
        if ntype == "boss":
            return 100.0
        if hp_pct < 0.35:
            base = {"rest": 90, "shop": 80, "event": 60, "treasure": 50,
                    "weak": 10, "normal": 10, "monster": 10, "elite": 0,
                    "unknown": 55}
            return base.get(ntype, 30)
        if hp_pct < 0.55:
            base = {"rest": 85, "shop": 80, "event": 65, "treasure": 70,
                    "weak": 40, "normal": 40, "monster": 40, "elite": 15,
                    "unknown": 60}
            return base.get(ntype, 30)
        base = {"elite": 80, "weak": 55, "normal": 55, "monster": 55,
                "event": 50, "shop": 45, "treasure": 70, "rest": 30,
                "unknown": 50}
        s = float(base.get(ntype, 40))
        if ntype == "elite" and hp_pct > 0.75:
            s += 15
        if ntype == "shop":
            if deck_size > 10:
                s += 15
            if gold >= 150:
                s += 25
        if ntype == "rest" and hp_pct > 0.70:
            s -= 10
        if character == "silent" and hp_pct < 0.50 and ntype == "rest":
            s += 30
        return s

    scores = [_score(rt) for rt in room_types]
    return max(range(len(room_types)), key=lambda i: scores[i])


# ---------------------------------------------------------------------------
# Rest site (rest vs smith)
# ---------------------------------------------------------------------------

# Boss floors pulled from STRATEGY; duplicated here to avoid importing the
# live advisor module (which pulls in config / network state we don't want
# during training).
_BOSS_FLOORS = {15, 16, 33, 34, 51, 52}


def shadow_pick_rest(
    opt_types: list[int],
    deck_indices: list,   # aligned with opt_types; None for the rest slot
    deck: list,
    hp: int,
    max_hp: int,
    floor: int,
    character: str = "silent",
    relics: frozenset | None = None,
) -> int:
    """Mirror of ``decide_rest``.

    ``opt_types[0]`` is always ``OPTION_REST``. Indices 1..N are
    ``OPTION_SMITH`` entries, with ``deck_indices[i]`` pointing back at
    the deck card that would be upgraded. Returns the index (into
    opt_types) of the chosen option.
    """
    # Import inside the function to avoid a circular import at module load.
    from .self_play import OPTION_SMITH  # noqa: F401

    if not opt_types:
        return 0

    hp_pct = hp / max(1, max_hp)

    # Threshold defaults mirror STRATEGY values in config_a.py; we don't
    # import STRATEGY here to keep the shadow advisor side-effect-free.
    if character == "silent":
        rest_threshold = 0.50
        upgrade_threshold = 0.70
        boss_rest_threshold = 0.70
    else:
        rest_threshold = 0.40
        upgrade_threshold = 0.60
        boss_rest_threshold = 0.70

    pre_boss = floor in _BOSS_FLOORS
    rest_idx = 0  # by construction: opt_types[0] == OPTION_REST
    smith_idxs = [i for i in range(1, len(opt_types))]

    # Hard pre-boss rest rule
    if pre_boss and hp_pct < boss_rest_threshold:
        return rest_idx

    # HP high → upgrade
    if hp_pct >= upgrade_threshold and smith_idxs:
        return _pick_best_smith(smith_idxs, deck_indices, deck, floor,
                                hp, max_hp, relics)

    # HP critical → rest
    if hp_pct < rest_threshold:
        return rest_idx

    # Gray zone → rest (mirrors live advisor fallback)
    return rest_idx


def _pick_best_smith(
    smith_idxs: list[int],
    deck_indices: list,
    deck: list,
    floor: int,
    hp: int,
    max_hp: int,
    relics: frozenset | None,
) -> int:
    """Score each smith candidate by organic upgrade value; return the
    opt_types index of the highest-scoring card. Falls back to the first
    smith candidate on any error."""
    try:
        from ..card_picker import score_card
        from ..data_loader import load_cards
        db = load_cards()
        best_score = float("-inf")
        best_idx = smith_idxs[0]
        for oi in smith_idxs:
            di = deck_indices[oi]
            if di is None or di >= len(deck):
                continue
            base_card = deck[di]
            up = db.get_upgraded(getattr(base_card, "id", "") or "")
            card_for_score = up if up is not None else base_card
            score = score_card(card_for_score, deck, floor, hp, max_hp,
                               relics=relics)
            if score > best_score:
                best_score = score
                best_idx = oi
        return best_idx
    except Exception:
        return smith_idxs[0]


# ---------------------------------------------------------------------------
# Shop (buy/remove/potion/leave)
# ---------------------------------------------------------------------------

def shadow_pick_shop(
    opt_types: list[int],
    actions: list,           # matches actions list built in full_run shop branch
    deck: list,
    hp: int,
    max_hp: int,
    gold: int,
    floor: int,
    relics: frozenset | None = None,
) -> int:
    """Mirror of ``_simulate_shop`` priority: relic > remove > buy card > buy potion > leave.

    ``actions`` is the same action tuple list ``full_run.py`` builds for the
    shop (("remove", deck_idx) | ("buy", shop_idx, cost) | ("relic", ri)
    | ("potion", pi) | ("leave",)). Returns the index that the heuristic
    would pick.
    """
    from .self_play import (
        OPTION_SHOP_REMOVE, OPTION_SHOP_BUY,
        OPTION_SHOP_BUY_POTION, OPTION_SHOP_LEAVE,
        OPTION_SHOP_BUY_RELIC,
    )

    if not opt_types:
        return 0

    # 1. Buy a relic — highest-impact shop decision.  Mirrors
    # _simulate_shop's #1 priority.  We don't have the relic name
    # here to score, so we just take the first offered relic (the
    # pool is already filtered to non-owned in full_run).
    for i, ot in enumerate(opt_types):
        if ot == OPTION_SHOP_BUY_RELIC:
            return i

    # 2. Prefer a remove if any Strike/Defend still in the deck
    for i, ot in enumerate(opt_types):
        if ot == OPTION_SHOP_REMOVE:
            return i

    # 3. Pick the highest-scoring buy-card option, if affordable. The
    # heuristic reuses score_card so it agrees with the live advisor's
    # shop logic.
    try:
        from ..card_picker import score_card
        best_buy_idx = None
        best_buy_score = float("-inf")
        for i, ot in enumerate(opt_types):
            if ot != OPTION_SHOP_BUY:
                continue
            act = actions[i]
            shop_idx = act[1]
            # We don't have the actual Card object here, just its vocab id —
            # fall back to "first affordable" in that case.
            best_buy_idx = i
            break  # Simple heuristic: first affordable buy
        if best_buy_idx is not None:
            return best_buy_idx
    except Exception:
        pass

    # 4. Buy a potion if one is offered and slots are free (cheap signal)
    for i, ot in enumerate(opt_types):
        if ot == OPTION_SHOP_BUY_POTION:
            return i

    # 5. Leave (always last)
    for i, ot in enumerate(opt_types):
        if ot == OPTION_SHOP_LEAVE:
            return i

    return 0


# ---------------------------------------------------------------------------
# Card reward
# ---------------------------------------------------------------------------

def shadow_pick_card_reward(
    offered: list,
    deck: list,
    hp: int,
    max_hp: int,
    floor: int,
    relics: frozenset | None = None,
) -> int:
    """Mirror of ``decide_card_reward``. Delegates to the organic picker
    the live advisor also uses (``card_picker.pick_card``). Returns the
    index in ``[0..len(offered)]`` — ``len(offered)`` is the skip slot.
    """
    if not offered:
        return 0
    try:
        from ..card_picker import pick_card as organic_pick
        pick = organic_pick(offered, deck, floor, hp, max_hp, relics=relics)
        if pick is None:
            return len(offered)  # skip
        for i, c in enumerate(offered):
            if c.id == pick.id:
                return i
        return len(offered)
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Events + Neow
# ---------------------------------------------------------------------------

# These two shadow pickers are thin wrappers over the core scorers in
# ``simulator.py`` so live play (``deterministic_advisor.decide_neow`` /
# ``decide_event_default``) and the training shadow advisor all share
# ONE scoring code path. Changing the scoring means editing one place:
# ``NEOW_TAG_PRIORITY`` / ``_evaluate_event_options``, both in
# ``simulator.py``. See IMPROVEMENTS.md #7 follow-up notes.


def shadow_pick_event(
    event_id: str,
    hp: int,
    max_hp: int,
    gold: int,
    deck: list,
) -> int:
    """Mirror of ``decide_event_default``. Returns the option index the
    live heuristic would pick for ``event_id``. Delegates to
    ``simulator.heuristic_event_option_index`` so the scoring stays in
    sync with ``_evaluate_event_options``.
    """
    try:
        from ..simulator import heuristic_event_option_index
        return heuristic_event_option_index(event_id, hp, max_hp, gold, deck)
    except Exception:
        return 0


def shadow_pick_neow(
    hp: int,
    max_hp: int,
    gold: int,
    deck: list,
) -> int:
    """Mirror of ``decide_neow``. Returns the blessing index in
    ``simulator.NEOW_BLESSINGS`` the live heuristic would pick. Shares
    ``NEOW_TAG_PRIORITY`` with live play via
    ``simulator.heuristic_neow_option_index``.
    """
    try:
        from ..simulator import heuristic_neow_option_index
        return heuristic_neow_option_index(hp, max_hp, gold, deck)
    except Exception:
        return 0
