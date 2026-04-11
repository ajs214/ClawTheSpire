"""Deterministic (rule-based) advisor for non-combat, non-event decisions.

Replaces LLM calls with codified strategy from config.py for:
- Rest sites (heal vs upgrade)
- Card rewards (tier-list + archetype matching)
- Map navigation (HP-threshold routing)
- Shop (auto-remove, tier-list buy, close)
- Boss relics (archetype-matched relic scoring)
- Deck select / upgrade (tier-list priority)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .config import (
    CARD_TIERS,
    CHARACTER_CONFIG,
    RELIC_GUIDE,
    STRATEGY,
    detect_character,
)
from .game_data import strip_markup

if TYPE_CHECKING:
    from .game_data import GameDataDB


# ---------------------------------------------------------------------------
# Lazy card-property lookup (loads CardDB once for property-based detection)
# ---------------------------------------------------------------------------

_CARD_PROPS_CACHE: dict[str, dict] | None = None


def _get_card_props() -> dict[str, dict]:
    """Lazily load card properties from card data.

    Returns a dict of {lowercase_name: {is_poison, is_shiv, is_sly, is_draw, is_defense}}
    built from card mechanical properties, not hardcoded lists.
    """
    global _CARD_PROPS_CACHE
    if _CARD_PROPS_CACHE is not None:
        return _CARD_PROPS_CACHE

    try:
        from .data_loader import load_cards
        from .card_picker import extract_properties
        db = load_cards()
        cache: dict[str, dict] = {}
        for card in db.all_cards():
            key = card.name.lower()
            if key in cache:
                continue
            props = extract_properties(card)
            cache[key] = {
                "is_poison": props.applies_poison > 0,
                "is_shiv": props.spawns_shivs,
                "is_sly": props.has_sly,
                "is_draw": props.draws_cards > 0,
                "is_defense": props.grants_block > 0 or props.grants_dexterity,
            }
        _CARD_PROPS_CACHE = cache
    except Exception:
        _CARD_PROPS_CACHE = {}
    return _CARD_PROPS_CACHE


def _card_prop(name: str, prop: str) -> bool:
    """Check a single property for a card by name."""
    return _get_card_props().get(name.lower(), {}).get(prop, False)


@dataclass
class Decision:
    action: str
    option_index: int | None
    reasoning: str
    network_value: float | None = None
    head_scores: dict | None = None
    # Decision-source telemetry (IMPROVEMENTS.md #11). Canonical values:
    #   - "advisor_tierlist":     rule-based decision from this module (default)
    #   - "organic_picker":       decide_card_reward's card_picker branch
    #   - "network_option_head":  runner._az_decide_* via pick_best_option
    #   - "mcts":                 combat turns driven by AlphaZeroMCTS
    #   - "fallback_first_action": _execute_deterministic's safe-default branch
    #   - "auto":                 collect_rewards_and_proceed / auto-action path
    # Read-only telemetry. Not used for dispatch — dispatch still uses
    # action/option_index/reasoning. Surfaces in run-log JSONL as the
    # `source` field on each `decision` event for later analysis.
    source: str = "advisor_tierlist"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_deck(state: dict) -> list[dict]:
    return state.get("run", {}).get("deck", [])


def _deck_names(state: dict) -> list[str]:
    """Return list of card names in deck (with + for upgrades)."""
    names = []
    for card in _get_deck(state):
        name = card.get("name", card.get("card_id", "?"))
        if card.get("upgraded"):
            name += "+"
        names.append(name)
    return names


def _deck_name_set(state: dict) -> set[str]:
    """Return set of base card names (no upgrade marker)."""
    return {card.get("name", card.get("card_id", "?")) for card in _get_deck(state)}


def _hp_pct(state: dict) -> float:
    run = state.get("run") or {}
    hp = run.get("current_hp", 0)
    max_hp = run.get("max_hp", 1)
    return hp / max_hp if max_hp > 0 else 1.0


def _floor(state: dict) -> int:
    return (state.get("run") or {}).get("floor", 0)


def _gold(state: dict) -> int:
    return (state.get("run") or {}).get("gold", 0)


def _get_relics(state: dict) -> frozenset[str]:
    """Extract the set of owned relic names from the game state.

    Relics live under run.relics.  Each entry may be a dict (with
    ``name`` / ``id`` keys) or a bare string.  We normalise to a
    frozenset of display names so the relic synergy layer can look
    them up by catalog name (e.g. "Snecko Skull", "Wrist Blade").
    """
    run = state.get("run") or {}
    raw = run.get("relics") or []
    names: set[str] = set()
    for r in raw:
        if isinstance(r, str):
            if r:
                names.add(r)
        elif isinstance(r, dict):
            nm = r.get("name") or r.get("id")
            if nm:
                names.add(nm)
    return frozenset(names)


def _card_tier(card_name: str, character: str, card_raw: dict | None = None) -> str:
    """Return tier (S/A/B/avoid) for a card.

    If the card isn't in the explicit tier list *and* a raw card dict is
    provided, fall back to ``_auto_tier_card`` which scores from the card's
    mechanical properties (damage, block, keywords, etc.).  Without a raw
    dict the fallback returns ``"B"`` so unlisted cards are never invisible.
    """
    tiers = CARD_TIERS.get(character, {})
    # Strip upgrade marker for lookup
    base = card_name.rstrip("+")
    for tier, cards in tiers.items():
        if base in cards:
            return tier
    # Fallback: auto-classify from card properties
    if card_raw is not None:
        return _auto_tier_card(card_raw, character)
    return "B"  # safe default for callers without raw dict


# ---------------------------------------------------------------------------
# Property-based auto-tier for cards not in the explicit tier list
# ---------------------------------------------------------------------------

# Keyword signals and their point values.  Positive = desirable.
_AUTO_TIER_KEYWORDS: list[tuple[str, float]] = [
    # High-value mechanics
    ("intangible",   30),
    ("retain",       6),
    ("innate",       4),
    ("sly",          3),
    ("exhaust",     -1),   # slight penalty — thins future draws but loses value
]

def _auto_tier_card(card: dict, character: str) -> str:
    """Score an unlisted card from its raw game-state dict and return a tier.

    Uses a simple point system based on mechanical properties — the same
    properties the 27-dim neural network encoding captures.  This keeps
    tier-list maintenance low: only truly novel / nuanced cards need manual
    placement; everything else gets a reasonable default.

    Scoring guide (roughly):
        >=70  → S     (exceptional standalone value)
        >=45  → A     (strong, would always consider)
        >=20  → B     (situational / average)
        <20   → avoid (actively bad or anti-synergy)
    """
    score = 15.0  # baseline: every card starts at low-B floor
    desc = (card.get("description") or "").lower()
    rarity = (card.get("rarity") or "").lower()
    card_type = (card.get("type") or "").lower()
    cost = card.get("energy_cost", card.get("cost", 1))
    if isinstance(cost, str):
        cost = -1  # X-cost
    damage = card.get("damage") or 0
    block = card.get("block") or 0
    keywords = [k.lower() for k in (card.get("keywords") or [])]

    # --- Rarity baseline ---
    if rarity == "rare":
        score += 15
    elif rarity == "uncommon":
        score += 5

    # --- Card type ---
    if card_type == "power":
        score += 10  # powers are inherently scaling

    # --- Cost efficiency ---
    if cost == 0:
        score += 12  # free cards are almost always good
    elif cost == -1:
        score += 8   # X-cost cards are flexible
    elif cost >= 3:
        score -= 8   # expensive cards are risky

    # --- Damage value ---
    if damage:
        # Normalize: 10 dmg for 1 energy is baseline (~0 bonus)
        efficiency = damage / max(cost, 1)
        if efficiency >= 15:
            score += 15
        elif efficiency >= 10:
            score += 8
        elif efficiency >= 7:
            score += 3

    # --- Block value ---
    if block:
        efficiency = block / max(cost, 1)
        if efficiency >= 12:
            score += 12
        elif efficiency >= 8:
            score += 6
        elif efficiency >= 5:
            score += 2

    # --- Hybrid (damage + block) bonus ---
    if damage and block:
        score += 10  # cards that do both are premium

    # --- Keyword signals ---
    for kw in keywords:
        for pattern, pts in _AUTO_TIER_KEYWORDS:
            if pattern in kw:
                score += pts

    # --- Description-based heuristics ---
    # Draw
    if "draw" in desc:
        # Count draw amount heuristically
        for n in (4, 3, 2):
            if f"draw {n}" in desc:
                score += n * 5
                break
        else:
            score += 4  # generic draw mention

    # Poison (Silent-specific)
    if character == "silent" and "poison" in desc:
        score += 8

    # Weak / Vulnerable application
    if "weak" in desc:
        score += 5
    if "vulnerable" in desc:
        score += 5

    # Energy gain
    if "energy" in desc and ("gain" in desc or "+" in desc):
        score += 8

    # Shiv generation
    if "shiv" in desc:
        score += 4

    # AoE / "all enemies"
    if "all enemies" in desc or "all enemy" in desc:
        score += 4

    # Strength reduction (defensive)
    if "strength" in desc and ("reduce" in desc or "lose" in desc or "-" in desc):
        score += 6

    # Negative signals
    if "lose hp" in desc or "lose max hp" in desc:
        score -= 6
    if "unplayable" in desc:
        score -= 20

    # --- Map to tier ---
    if score >= 70:
        return "S"
    elif score >= 45:
        return "A"
    elif score >= 20:
        return "B"
    else:
        return "avoid"


_TIER_RANK = {"S": 0, "A": 1, "B": 2, "avoid": 99}


def _detect_archetype(state: dict, character: str) -> tuple[str | None, set[str]]:
    """Detect the dominant deck archetype from card properties. Returns (name, matching_cards)."""
    deck_names = _deck_name_set(state)

    if character == "silent":
        archetype_prop = {"Shiv": "is_shiv", "Poison": "is_poison", "Sly": "is_sly"}
    else:  # ironclad — keep name-based for now (no property extraction for IC archetypes yet)
        archetypes = {
            "Strength": {"Inflame", "Demon Form", "Limit Break"},
            "Exhaust": {"Feel No Pain", "Corruption", "Dark Embrace"},
            "Block": {"Barricade", "Body Slam"},
        }
        best_name, best_count, best_cards = None, 0, set()
        for name, cards in archetypes.items():
            overlap = deck_names & cards
            if len(overlap) > best_count:
                best_name, best_count, best_cards = name, len(overlap), overlap
        return (best_name, best_cards) if best_count > 0 else (None, set())

    # Silent: property-based detection
    best_name, best_count, best_cards = None, 0, set()
    for arch_name, prop_key in archetype_prop.items():
        matching = {n for n in deck_names if _card_prop(n, prop_key)}
        if len(matching) > best_count:
            best_name, best_count, best_cards = arch_name, len(matching), matching

    return (best_name, best_cards) if best_count > 0 else (None, set())


def _is_defense_card(card_name: str, character: str) -> bool:
    """Check if a card is a dedicated block/defense card (property-based)."""
    return _card_prop(card_name, "is_defense")


def _is_in_archetype(card_name: str, archetype: str | None, character: str) -> bool:
    """Check if a card fits the current archetype (property-based for Silent)."""
    if archetype is None:
        return True  # No archetype yet, anything goes

    # Property-based matching for Silent archetypes
    archetype_to_prop = {
        "Shiv": "is_shiv", "Poison": "is_poison", "Sly": "is_sly",
    }
    prop_key = archetype_to_prop.get(archetype)
    if prop_key and _card_prop(card_name, prop_key):
        return True

    # Ironclad: name-based fallback (no property extraction for IC archetypes yet)
    if character == "ironclad":
        ic_archetypes = {
            "Strength": {"Inflame", "Demon Form", "Limit Break",
                         "Twin Strike", "Thrash", "Whirlwind", "Pommel Strike", "Brand"},
            "Exhaust": {"Feel No Pain", "Corruption", "Dark Embrace", "Burning Pact",
                        "True Grit", "Offering"},
            "Block": {"Barricade", "Body Slam", "Shrug It Off",
                      "Impervious", "Flame Barrier", "Juggernaut"},
        }
        if card_name in ic_archetypes.get(archetype, set()):
            return True

    # Defense and draw cards are always acceptable
    if _is_defense_card(card_name, character):
        return True
    if _card_prop(card_name, "is_draw"):
        return True
    return False


def _relic_matches_archetype(relic_name: str, archetype: str | None, character: str) -> float:
    """Score a relic 0-2 based on archetype fit. 2=top_pick, 1=archetype match, 0=no match."""
    guide = RELIC_GUIDE.get(character, {})

    # Check top_picks first (always good)
    top = guide.get("top_picks", {}).get("relics", [])
    if relic_name in top:
        return 2.0

    # Check avoid list
    avoid = guide.get("avoid", {}).get("relics", [])
    if relic_name in avoid:
        return -1.0

    # Check archetype-specific categories
    archetype_to_category = {
        "Strength": "strength_scaling",
        "Exhaust": "exhaust_engine",
        "Block": "block_build",
        "Shiv": "shiv_synergy",
        "Poison": "poison_synergy",
        "Sly": "sly_synergy",
    }

    if archetype:
        cat_key = archetype_to_category.get(archetype)
        if cat_key:
            cat_relics = guide.get(cat_key, {}).get("relics", [])
            if relic_name in cat_relics:
                return 1.5

    # Check all non-avoid categories for a partial match
    for key, info in guide.items():
        if key in ("top_picks", "avoid"):
            continue
        if relic_name in info.get("relics", []):
            return 0.5

    return 0.0


# ---------------------------------------------------------------------------
# Rest site
# ---------------------------------------------------------------------------

def decide_rest(state: dict) -> Decision:
    """Deterministic rest site decision: heal vs upgrade."""
    character = detect_character(state)
    hp_pct = _hp_pct(state)
    floor = _floor(state)

    rest_data = state.get("rest") or {}
    if not rest_data:
        rest_data = (state.get("agent_view") or {}).get("rest") or {}
    options = rest_data.get("options", [])

    # Find rest (heal) and upgrade option indices
    rest_idx, upgrade_idx = None, None
    for i, opt in enumerate(options):
        name = (opt.get("name") or opt.get("title") or opt.get("id", "")).lower()
        idx = opt.get("index", i)
        if "rest" in name or "heal" in name or "sleep" in name:
            rest_idx = idx
        elif "upgrade" in name or "smith" in name:
            upgrade_idx = idx

    # Thresholds — read from the active config profile so A/B can actually
    # differ on rest strategy. Historically these were hardcoded constants,
    # which meant the STRATEGY values were dead code (see IMPROVEMENTS.md #1).
    # Silent is still the only character we actively tune, so non-Silent
    # characters fall back to slightly lower thresholds.
    if character == "silent":
        rest_threshold = STRATEGY.get("rest_heal_threshold", 0.50)
        upgrade_threshold = STRATEGY.get("rest_upgrade_threshold", 0.70)
        boss_rest_threshold = STRATEGY.get("boss_rest_threshold", 0.70)
    else:
        rest_threshold = STRATEGY.get("rest_heal_threshold", 0.40)
        upgrade_threshold = STRATEGY.get("rest_upgrade_threshold", 0.60)
        boss_rest_threshold = STRATEGY.get("boss_rest_threshold", 0.70)
    pre_boss = floor in STRATEGY.get("boss_floors", set())

    # Decision logic
    if pre_boss and hp_pct < boss_rest_threshold and rest_idx is not None:
        return Decision("choose_rest_option", rest_idx,
                        f"Pre-boss heal (HP {hp_pct:.0%} < {boss_rest_threshold:.0%})")

    if hp_pct > 0.80 and upgrade_idx is not None:
        # Find best card to upgrade (done by the game UI, we just pick upgrade)
        return Decision("choose_rest_option", upgrade_idx,
                        f"HP high ({hp_pct:.0%}), upgrading")

    if hp_pct >= upgrade_threshold and upgrade_idx is not None:
        return Decision("choose_rest_option", upgrade_idx,
                        f"HP decent ({hp_pct:.0%}), prefer upgrade")

    if hp_pct < rest_threshold and rest_idx is not None:
        return Decision("choose_rest_option", rest_idx,
                        f"HP critical ({hp_pct:.0%}), must rest")

    # Gray zone: check if we have un-upgraded S/A-tier cards
    if upgrade_idx is not None:
        deck = _get_deck(state)
        has_upgradeable_key = any(
            not card.get("upgraded")
            and _card_tier(card.get("name", ""), character, card) in ("S", "A")
            for card in deck
        )
        if has_upgradeable_key:
            return Decision("choose_rest_option", upgrade_idx,
                            f"HP mid ({hp_pct:.0%}), have key cards to upgrade")

    # Default: rest if available, else upgrade, else first option
    if rest_idx is not None:
        return Decision("choose_rest_option", rest_idx,
                        f"HP mid ({hp_pct:.0%}), defaulting to rest")
    if upgrade_idx is not None:
        return Decision("choose_rest_option", upgrade_idx,
                        "No rest option, upgrading")
    # Fallback
    idx = options[0].get("index", 0) if options else 0
    return Decision("choose_rest_option", idx, "Only option available")


# ---------------------------------------------------------------------------
# Card reward
# ---------------------------------------------------------------------------

def decide_card_reward(state: dict, game_data: GameDataDB) -> Decision:
    """Card reward decision using the organic card picker.

    Primary path: build Card objects from game state and use the same
    score_card() / score_skip() logic that the simulator uses.  This
    ensures training and live play evaluate cards identically.

    Fallback path: tier-list scoring if card DB is unavailable.
    """
    character = detect_character(state)
    deck = _get_deck(state)
    deck_size = len(deck)
    deck_names = _deck_name_set(state)
    archetype, _ = _detect_archetype(state, character)
    floor = _floor(state)
    run = state.get("run") or {}
    hp = run.get("current_hp", 50)
    max_hp = run.get("max_hp", 80)
    relics = _get_relics(state)

    # Extract card options from game state
    reward = state.get("reward") or state.get("selection") or {}
    cards = reward.get("cards") or reward.get("card_options") or []
    if not cards:
        sel = state.get("selection") or {}
        cards = sel.get("cards", [])
    if not cards:
        cards = ((state.get("agent_view") or {}).get("reward") or {}).get("cards", [])

    if not cards:
        return Decision("skip_reward_cards", None, "No card options available")

    actions = state.get("available_actions", [])

    # --- Primary path: organic card picker (rule-based) ---
    deck_objs = _build_deck_card_objects(state)
    if deck_objs:
        try:
            from .card_picker import score_card, score_skip

            skip_score = score_skip(deck_objs, floor)
            best_idx, best_score, best_name = None, -1.0, ""

            for i, card in enumerate(cards):
                card_name = card.get("name", card.get("card_id", "?"))
                idx = card.get("index", i)
                card_obj = _resolve_card_obj(card_name)
                if not card_obj:
                    continue
                card_score = score_card(
                    card_obj, deck_objs, floor, hp, max_hp, relics=relics)
                if card_score > best_score:
                    best_idx = idx
                    best_score = card_score
                    best_name = card_name

            # Pick if best card beats skip threshold
            if best_score > skip_score and best_idx is not None:
                if "choose_reward_card" in actions:
                    return Decision(
                        "choose_reward_card", best_idx,
                        f"Taking {best_name} (score={best_score:.2f}, "
                        f"skip={skip_score:.2f})",
                        source="organic_picker")

            # Skip — no card worth taking
            if "skip_reward_cards" in actions:
                reason = (f"Skipping (best: {best_name}={best_score:.2f}, "
                          f"skip={skip_score:.2f})")
                return Decision("skip_reward_cards", None, reason,
                                source="organic_picker")

            # Fallback if neither action available
            return Decision("skip_reward_cards", None, "No valid action",
                            source="organic_picker")

        except Exception:
            pass  # Fall through to tier-list fallback

    # --- Fallback path: tier-list scoring ---
    has_defense = bool({c for c in deck_names if _is_defense_card(c, character)})

    best_idx, best_score, best_name, best_reason = None, -999, "", ""

    for i, card in enumerate(cards):
        card_name = card.get("name", card.get("card_id", "?"))
        idx = card.get("index", i)
        tier = _card_tier(card_name, character, card)

        if tier == "S":
            score = 100
        elif tier == "A":
            score = 70
        elif tier == "B":
            score = 30
        elif tier == "avoid":
            score = -50
        else:
            score = 15  # Unknown cards get a low base

        # Archetype bonuses/penalties
        enforce_archetype = (character == "silent" and floor >= 5
                             and archetype is not None)
        if archetype is not None:
            if _is_in_archetype(card_name, archetype, character):
                score += 20
            elif enforce_archetype:
                score -= 60

        # Defense card bonus if deck has none
        if not has_defense and floor >= 4 and _is_defense_card(card_name, character):
            score += 30

        # Relaxed deck size penalties (aligned with organic picker)
        if deck_size >= 18:
            if tier not in ("S",):
                score -= 50
        elif deck_size >= 16:
            if tier not in ("S", "A"):
                score -= 30

        if score > best_score:
            best_idx = idx
            best_score = score
            best_name = card_name
            best_reason = f"{card_name} (tier={tier or '?'})"

    # Relaxed skip thresholds (aligned with organic picker)
    skip_threshold = 30
    if deck_size >= 18:
        skip_threshold = 70
    elif deck_size >= 16:
        skip_threshold = 55
    elif deck_size >= 14:
        skip_threshold = 40

    if best_score < skip_threshold and "skip_reward_cards" in actions:
        return Decision("skip_reward_cards", None,
                        f"No good options (best: {best_reason}, score={best_score})")

    if best_idx is not None and "choose_reward_card" in actions:
        return Decision("choose_reward_card", best_idx,
                        f"Taking {best_reason} (score={best_score})")

    if "skip_reward_cards" in actions:
        return Decision("skip_reward_cards", None, "Skipping card reward")
    return Decision("skip_reward_cards", None, "No valid action")


# ---------------------------------------------------------------------------
# Map navigation
# ---------------------------------------------------------------------------

def decide_map(state: dict) -> Decision:
    """Deterministic map navigation: HP-threshold routing."""
    character = detect_character(state)
    hp_pct = _hp_pct(state)
    deck_size = len(_get_deck(state))
    gold = _gold(state)
    floor = _floor(state)

    map_data = state.get("map") or {}
    if not map_data:
        map_data = (state.get("agent_view") or {}).get("map") or {}
    nodes = map_data.get("available_nodes") or map_data.get("nodes") or []

    if not nodes:
        return Decision("choose_map_node", 0, "No node data, picking first")

    # Classify nodes
    def _node_type(node: dict) -> str:
        t = (node.get("node_type") or node.get("type") or
             node.get("icon") or node.get("symbol", "")).lower()
        if "elite" in t:
            return "elite"
        if "boss" in t:
            return "boss"
        if "rest" in t:
            return "rest"
        if "shop" in t or "merchant" in t:
            return "shop"
        if "event" in t or "unknown" in t or "mystery" in t:
            return "event"
        if "treasure" in t or "chest" in t:
            return "treasure"
        if "monster" in t or "enemy" in t or "combat" in t:
            return "monster"
        return "unknown"

    typed_nodes = []
    for i, node in enumerate(nodes):
        idx = node.get("index", i)
        ntype = _node_type(node)
        typed_nodes.append((idx, ntype))

    # Score each node based on current state
    def _score_node(idx: int, ntype: str) -> tuple[float, str]:
        if ntype == "boss":
            return (100.0, "boss (must go)")  # No choice usually

        if hp_pct < 0.35:
            # Critical HP: rest > shop > event > everything else
            scores = {"rest": 90, "shop": 80, "event": 60, "treasure": 50,
                      "monster": 10, "elite": 0, "unknown": 55}
            return (scores.get(ntype, 30), f"HP critical ({hp_pct:.0%})")

        if hp_pct < 0.55:
            # Low HP: avoid elites, prefer safe nodes
            scores = {"rest": 85, "shop": 80, "event": 65, "treasure": 70,
                      "monster": 40, "elite": 15, "unknown": 60}
            s = scores.get(ntype, 30)
            return (s, f"HP low ({hp_pct:.0%})")

        # Healthy: score based on value
        scores = {"elite": 80, "monster": 55, "event": 50, "shop": 45,
                  "treasure": 70, "rest": 30, "unknown": 50}
        s = scores.get(ntype, 40)

        # Elite bonus when HP is high
        if ntype == "elite" and hp_pct > 0.75:
            s += 15

        # Shop bonus when deck is large or gold is high
        if ntype == "shop":
            if deck_size > 10:
                s += 15
            if gold >= 150:
                s += 25

        # Rest penalty when HP is high (don't waste it)
        if ntype == "rest" and hp_pct > 0.70:
            s -= 10

        # Silent-specific: push rest when HP < 50%
        if character == "silent" and hp_pct < 0.50 and ntype == "rest":
            s += 30

        return (s, f"HP {hp_pct:.0%}, gold={gold}")

    scored = [(idx, ntype, *_score_node(idx, ntype)) for idx, ntype in typed_nodes]
    scored.sort(key=lambda x: x[2], reverse=True)

    best_idx, best_type, best_score, reason = scored[0]
    return Decision("choose_map_node", best_idx,
                    f"{best_type} node ({reason})")


# ---------------------------------------------------------------------------
# Shop
# ---------------------------------------------------------------------------

def _resolve_card_obj(card_name: str):
    """Resolve a card name to a Card object from the card DB.

    Returns None if the card can't be found.  Used to bridge live game
    state (dict with name strings) to the organic scorer (needs Card objects).
    """
    try:
        from .data_loader import load_cards
        db = load_cards()
        # Try exact match, then case-insensitive search
        for card in db.all_cards():
            if card.name == card_name or card.name == card_name.rstrip("+"):
                return card
    except Exception:
        pass
    return None


def _build_deck_card_objects(state: dict) -> list:
    """Convert the deck from game state dicts to Card objects.

    Falls back to an empty list if the card DB isn't available.
    """
    try:
        from .data_loader import load_cards
        db = load_cards()
        deck_cards = []
        for card_dict in _get_deck(state):
            name = card_dict.get("name", card_dict.get("card_id", "?"))
            card_id = card_dict.get("card_id") or card_dict.get("id", "")
            upgraded = card_dict.get("upgraded", False)

            # Try ID first, then name search
            card_obj = None
            if card_id:
                if upgraded:
                    card_obj = db.get_upgraded(card_id)
                if not card_obj:
                    card_obj = db.get(card_id)
            if not card_obj:
                card_obj = _resolve_card_obj(name)
            if card_obj:
                deck_cards.append(card_obj)
        return deck_cards
    except Exception:
        return []


# Protected cards that should never be removed from the deck
_LIVE_PROTECTED_CARDS = frozenset({
    "Survivor", "Neutralize", "Bash", "Eruption", "Vigilance",
})


def decide_shop(state: dict, game_data: GameDataDB) -> Decision:
    """Deterministic shop: smart remove > buy archetype relic > buy card > buy potion > close.

    Uses the organic card picker scoring to evaluate card removal targets
    and purchase candidates, matching the simulator's improved logic.
    """
    character = detect_character(state)
    actions = state.get("available_actions", [])
    deck_size = len(_get_deck(state))
    deck_names = _deck_name_set(state)
    gold = _gold(state)
    archetype, _ = _detect_archetype(state, character)
    floor = _floor(state)
    run = state.get("run") or {}
    hp = run.get("current_hp", 50)
    max_hp = run.get("max_hp", 80)
    owned_relics = _get_relics(state)

    shop = state.get("shop") or {}
    if not shop:
        shop = (state.get("agent_view") or {}).get("shop") or {}

    # Build Card objects for the organic scorer
    deck_objs = _build_deck_card_objects(state)

    # Priority 1: Remove the weakest card (not just Strike/Defend)
    if "remove_card_at_shop" in actions:
        remove_cost = shop.get("remove_cost", 75)
        if isinstance(remove_cost, int) and remove_cost <= gold and deck_size >= 8:
            best_remove_name = None
            best_remove_score = 999.0

            if deck_objs:
                # Use the organic scorer — lower score = weaker card
                try:
                    from .card_picker import score_card, extract_properties, \
                        build_signature, _card_power_score, _alignment_score
                    for card_obj in deck_objs:
                        if card_obj.name in _LIVE_PROTECTED_CARDS:
                            continue
                        props = extract_properties(card_obj)
                        sig = build_signature(deck_objs)
                        power = _card_power_score(card_obj, props)
                        # Halve alignment penalty (card is already in deck)
                        alignment = _alignment_score(card_obj, props, sig) * 0.5
                        upgrade_bonus = 0.05 if card_obj.upgraded else 0.0
                        removal_score = max(0.01, power + alignment + upgrade_bonus)
                        if removal_score < best_remove_score:
                            best_remove_score = removal_score
                            best_remove_name = card_obj.name
                except Exception:
                    deck_objs = []  # Fall back below

            if not deck_objs:
                # Fallback: Strike > Defend > nothing
                if "Strike" in deck_names:
                    best_remove_name = "Strike"
                    best_remove_score = 0.01
                elif "Defend" in deck_names:
                    best_remove_name = "Defend"
                    best_remove_score = 0.01

            # Only remove if the card is genuinely weak
            is_basic = best_remove_name in ("Strike", "Defend")
            if best_remove_name and (best_remove_score < 0.25 or is_basic):
                return Decision("remove_card_at_shop", None,
                                f"Removing {best_remove_name} "
                                f"(value={best_remove_score:.2f}, {remove_cost}g)")

    # Priority 2: Buy a relic that matches the *actual* deck composition
    if "buy_relic" in actions:
        shop_relics = shop.get("relics", [])
        best_relic_idx, best_relic_score, best_relic_name = None, 0.0, ""
        # Prefer the deck-aware scorer; fall back to archetype match if
        # relic_synergy or deck_objs aren't available.
        deck_aware_scorer = None
        try:
            from .relic_synergy import score_relic_for_deck
            deck_aware_scorer = score_relic_for_deck
        except Exception:
            deck_aware_scorer = None

        for i, relic in enumerate(shop_relics):
            price = relic.get("price", relic.get("cost", 999))
            if not isinstance(price, int) or price > gold:
                continue
            name = relic.get("name", relic.get("id", "?"))
            if deck_aware_scorer is not None and deck_objs:
                score = deck_aware_scorer(name, deck_objs)
            else:
                score = _relic_matches_archetype(name, archetype, character)
            if score > best_relic_score:
                best_relic_idx = i
                best_relic_score = score
                best_relic_name = name

        # Only buy relics that are top picks or archetype matches
        if best_relic_score >= 1.0 and best_relic_idx is not None:
            return Decision("buy_relic", best_relic_idx,
                            f"Buying {best_relic_name} (deck fit={best_relic_score:.2f})")

    # Priority 3: Buy a card (organic scorer, relaxed deck-size threshold)
    if "buy_card" in actions:
        cards = shop.get("cards", [])
        best_card_idx, best_card_score, best_card_name = None, -1.0, ""

        if deck_objs:
            # Use organic scorer for each affordable shop card
            try:
                from .card_picker import score_card, score_skip
                skip_score = score_skip(deck_objs, floor)

                for i, card in enumerate(cards):
                    price = card.get("price", card.get("cost", 999))
                    if not isinstance(price, int) or price > gold:
                        continue
                    name = card.get("name", card.get("id", "?"))
                    card_obj = _resolve_card_obj(name)
                    if not card_obj:
                        continue
                    card_score = score_card(
                        card_obj, deck_objs, floor, hp, max_hp,
                        relics=owned_relics)
                    # Must beat skip threshold to be worth buying
                    if card_score > skip_score and card_score > best_card_score:
                        best_card_idx = i
                        best_card_score = card_score
                        best_card_name = name
            except Exception:
                deck_objs = []  # Fall back below

        if not deck_objs:
            # Fallback: tier-list based (original logic but with relaxed cap)
            if deck_size < 18:
                for i, card in enumerate(cards):
                    price = card.get("price", card.get("cost", 999))
                    if not isinstance(price, int) or price > gold:
                        continue
                    name = card.get("name", card.get("id", "?"))
                    tier = _card_tier(name, character, card)
                    if tier not in ("S", "A"):
                        continue
                    if archetype and not _is_in_archetype(name, archetype, character):
                        continue
                    card_score = 100 - _TIER_RANK.get(tier, 99) * 30
                    if card_score > best_card_score:
                        best_card_idx = i
                        best_card_score = card_score
                        best_card_name = name

        if best_card_idx is not None and best_card_score > 0:
            return Decision("buy_card", best_card_idx,
                            f"Buying {best_card_name} (score={best_card_score:.2f})")

    # Priority 4: Buy a potion if HP is low and we have room
    if "buy_potion" in actions:
        hp_ratio = hp / max(1, max_hp)
        if hp_ratio < 0.55 and gold >= 50:
            potions = shop.get("potions", [])
            # Prefer heal potions when low HP, otherwise pick cheapest useful one
            best_pot_idx, best_pot_name = None, ""
            for i, pot in enumerate(potions):
                price = pot.get("price", pot.get("cost", 999))
                if not isinstance(price, int) or price > gold:
                    continue
                name = pot.get("name", pot.get("id", "?"))
                # Prioritise healing potions when HP is critical
                if hp_ratio < 0.35 and "blood" in name.lower():
                    best_pot_idx = i
                    best_pot_name = name
                    break
                if best_pot_idx is None:
                    best_pot_idx = i
                    best_pot_name = name

            if best_pot_idx is not None:
                return Decision("buy_potion", best_pot_idx,
                                f"Buying {best_pot_name} (HP at {hp_ratio:.0%})")

    # Priority 5: Close shop
    if "close_shop_inventory" in actions:
        return Decision("close_shop_inventory", None,
                        "Nothing worth buying, leaving shop")

    # Fallback
    return Decision("close_shop_inventory", None, "Done shopping")


# ---------------------------------------------------------------------------
# Neow event (starting bonus)
# ---------------------------------------------------------------------------

# IMPROVEMENTS.md #7 follow-up: Neow scoring is now shared between live
# play (this module) and training (``simulator.heuristic_neow_option_index``)
# via ``simulator.score_neow_option`` + ``NEOW_TAG_PRIORITY``. The old
# ``_NEOW_KEYWORD_SCORES`` table that used to live here has been replaced
# by ``simulator._NEOW_TEXT_KEYWORDS`` (keyword → tag) and
# ``simulator.NEOW_TAG_PRIORITY`` (tag → score). If you want to tweak how
# Neow options are ranked, edit those two tables — both scorers will
# pick it up.


def decide_neow(state: dict) -> Decision | None:
    """Deterministic Neow (starting bonus) picker.

    Returns a Decision if the current screen is the Neow event, or None
    if this isn't a Neow event (so the caller can fall through to the
    LLM for other events).
    """
    from .simulator import (
        classify_neow_option_text,
        score_neow_option,
        _NEOW_TEXT_KEYWORDS,
    )

    event = state.get("event") or {}
    if not event:
        event = (state.get("agent_view") or {}).get("event") or {}

    # Only handle Neow — detect by event name or floor-1 heuristic
    event_name = (event.get("name") or event.get("event_id") or "").lower()
    floor = _floor(state)
    is_neow = "neow" in event_name or (floor <= 1 and event.get("options"))
    if not is_neow:
        return None

    # Guard: after picking a Neow bonus like "Scroll Boxes", the game may
    # show a follow-up screen (e.g. "Choose a Pack") that is still an event
    # on floor 1.  Detect this by checking whether any option text matches
    # known Neow keywords.  If none match, this isn't the Neow menu — bail
    # so the LLM or a generic handler can deal with it.
    options = event.get("options") or []
    _all_text = " ".join(
        ((o.get("name") or "") + " " + (o.get("description") or "")).lower()
        for o in options
    )
    _has_neow_signal = any(kw in _all_text for kw, _tag in _NEOW_TEXT_KEYWORDS)
    if not _has_neow_signal and "neow" not in event_name:
        # This is a Neow follow-up sub-screen (e.g. "Choose a Pack").
        # Auto-pick option 0 so we don't get stuck.
        if floor <= 1 and options:
            first_name = options[0].get("name") or options[0].get("title") or "option 0"
            return Decision("choose_event_option", 0,
                            f"Neow sub-screen: pick {first_name}")
        return None

    if not options:
        return None

    # Current HP fraction gates conditional tags (full_heal, risky_trade).
    run = state.get("run") or {}
    hp = int(run.get("current_hp") or 0)
    max_hp = int(run.get("max_hp") or 1)
    hp_frac = hp / max_hp if max_hp > 0 else 1.0

    best_idx, best_score, best_reason = 0, -999.0, "fallback"

    for opt in options:
        idx = opt.get("index", 0)
        # Build a combined text from all available fields
        name = opt.get("name") or opt.get("title") or ""
        desc = opt.get("description") or opt.get("desc") or ""
        text = f"{name} — {desc}"

        # Classify against the shared keyword table, then score via the
        # shared tag priority + HP-penalty code path.
        tag = classify_neow_option_text(text)
        score = score_neow_option(tag=tag, text=text, hp_frac=hp_frac)

        if score > best_score:
            best_idx = idx
            best_score = score
            best_reason = f"{name} ({tag}, score={score:.1f})"

    return Decision("choose_event_option", best_idx,
                     f"Neow: {best_reason}")


# ---------------------------------------------------------------------------
# Non-Neow event default
# ---------------------------------------------------------------------------

def decide_event_default(state: dict) -> Decision | None:
    """Deterministic Floor 2+ event picker.

    Reuses the simulator's ``_evaluate_event_options`` scorer so that live-play
    event decisions match the canned outcomes the training loop assumed for
    the same event. This is the key invariant: if the training simulator
    decided a run's outcome assuming Wood Carvings picks Bird, live play must
    also pick Bird or the value-head targets are wrong.

    Returns a ``Decision`` to call ``choose_event_option``, or ``None`` if
    the event screen state is unreadable (caller should fall back).
    """
    event = state.get("event") or {}
    if not event:
        event = (state.get("agent_view") or {}).get("event") or {}

    # Keep only unlocked/available options but preserve the original game
    # indices so we can hand the correct one back to the game client.
    raw_options = event.get("options") or []
    options = [o for o in raw_options if not o.get("locked")]
    if not options:
        return None

    run = state.get("run") or {}
    hp = int(run.get("current_hp") or 0)
    max_hp = int(run.get("max_hp") or 1)
    gold = int(run.get("gold") or 0)
    deck = _get_deck(state)  # unused by the scorer today, but keeps the
                             # signature aligned in case the sim grows
                             # deck-aware event logic later.

    try:
        from .simulator import _evaluate_event_options
    except Exception:
        return None

    best = _evaluate_event_options(options, hp, max_hp, gold, deck)
    if not best:
        return None

    # Prefer the option's own ``index`` field (what the game client expects);
    # fall back to its position in the live options list.
    chosen_idx = best.get("index")
    if chosen_idx is None:
        try:
            chosen_idx = raw_options.index(best)
        except ValueError:
            chosen_idx = 0

    event_name = (
        event.get("name")
        or event.get("event_id")
        or event.get("id")
        or "event"
    )
    opt_name = (
        best.get("name")
        or best.get("title")
        or best.get("description", "")[:40]
        or f"option {chosen_idx}"
    )
    return Decision(
        "choose_event_option",
        int(chosen_idx),
        f"{event_name}: {opt_name} (sim scorer)",
    )


# ---------------------------------------------------------------------------
# Boss relic
# ---------------------------------------------------------------------------

def decide_boss_relic(state: dict, game_data: GameDataDB) -> Decision:
    """Deterministic boss relic pick: score against the actual deck."""
    character = detect_character(state)
    archetype, _ = _detect_archetype(state, character)
    deck_objs = _build_deck_card_objects(state)

    # Find relic options
    chest = state.get("chest") or {}
    reward = state.get("reward") or state.get("selection") or {}
    relic_options = chest.get("relics", []) or reward.get("relics", [])
    if not relic_options:
        relic_options = ((state.get("agent_view") or {}).get("chest") or {}).get("relics", [])

    if not relic_options:
        return Decision("choose_treasure_relic", 0, "No relic data, picking first")

    # Prefer the deck-aware relic scorer; fall back to archetype fit.
    deck_aware_scorer = None
    try:
        from .relic_synergy import score_relic_for_deck
        deck_aware_scorer = score_relic_for_deck
    except Exception:
        deck_aware_scorer = None

    best_idx, best_score, best_name = 0, -999.0, ""
    for i, relic in enumerate(relic_options):
        name = relic.get("name", relic.get("relic_id", relic.get("id", "?")))
        idx = relic.get("index", i)
        if deck_aware_scorer is not None and deck_objs:
            score = deck_aware_scorer(name, deck_objs)
        else:
            score = _relic_matches_archetype(name, archetype, character)
        if score > best_score:
            best_idx = idx
            best_score = score
            best_name = name

    return Decision("choose_treasure_relic", best_idx,
                    f"{best_name} (archetype fit={best_score})")


# ---------------------------------------------------------------------------
# Deck select (upgrade / remove / transform)
# ---------------------------------------------------------------------------

def _organic_removal_score(card_obj, deck_objs) -> float:
    """Lower score = better removal candidate (shares the simulator formula).

    Uses intrinsic power + halved archetype alignment penalty.  Matches
    simulator._score_card_for_removal and the shop-removal path in
    decide_shop so every 'remove a card' surface agrees.
    """
    try:
        from .card_picker import (
            extract_properties, build_signature,
            _card_power_score, _alignment_score,
        )
        props = extract_properties(card_obj)
        sig = build_signature(deck_objs)
        power = _card_power_score(card_obj, props)
        alignment = _alignment_score(card_obj, props, sig) * 0.5
        upgrade_bonus = 0.05 if getattr(card_obj, "upgraded", False) else 0.0
        return max(0.01, power + alignment + upgrade_bonus)
    except Exception:
        return 0.50


def _organic_upgrade_value(card_obj, deck_objs, floor: int, hp: int, max_hp: int,
                           relics: frozenset[str] | set[str] | None = None) -> float:
    """Score how valuable it is to upgrade this card.

    Uses score_card on the upgraded version (if resolvable), plus a stat
    delta kicker.  Falls back to raw power score.  Higher = better upgrade
    target.  Matches the spirit of simulator._rest_site_decision.
    """
    try:
        from .card_picker import score_card, extract_properties, _card_power_score
        from .data_loader import load_cards

        db = load_cards()
        base_id = getattr(card_obj, "id", "") or ""
        upgraded = db.get_upgraded(base_id) if base_id else None

        base_props = extract_properties(card_obj)
        base_power = _card_power_score(card_obj, base_props)

        if upgraded is not None:
            # Value = score of the upgraded card in this deck.
            value = score_card(upgraded, deck_objs, floor, hp, max_hp,
                               relics=relics)
            up_props = extract_properties(upgraded)
            # Stat-delta kicker (helps break ties with clearer upgrades)
            if up_props.deals_damage > base_props.deals_damage:
                value += (up_props.deals_damage - base_props.deals_damage) * 0.02
            if up_props.grants_block > base_props.grants_block:
                value += (up_props.grants_block - base_props.grants_block) * 0.02
            if up_props.draws_cards > base_props.draws_cards:
                value += (up_props.draws_cards - base_props.draws_cards) * 0.10
            if up_props.applies_poison > base_props.applies_poison:
                value += (up_props.applies_poison - base_props.applies_poison) * 0.03
            return value

        # No upgraded variant available — fall back to power of the base card.
        return base_power
    except Exception:
        return 0.0


def decide_deck_select(state: dict) -> Decision:
    """Deterministic deck card selection for upgrade/remove/transform.

    Uses the organic scorer from card_picker for remove/transform/discard
    and the upgrade-value helper for upgrades, so every deck-edit surface
    (shop removal, event removal, rest site upgrades) agrees on what a
    card is worth in the current deck's archetype.
    """
    character = detect_character(state)
    cfg = CHARACTER_CONFIG.get(character, CHARACTER_CONFIG["ironclad"])
    protect_cards = set(cfg.get("protect_cards", [cfg["key_card"]]))

    sel = state.get("selection") or {}
    prompt = strip_markup(sel.get("prompt") or "").lower()
    cards = sel.get("cards", [])

    if not cards:
        return Decision("select_deck_card", 0, "No cards to choose from")

    is_remove = "remove" in prompt
    # "smith" is the STS2 rest-site upgrade prompt — treat it as an
    # upgrade screen so the upgrade-value scorer fires instead of the
    # generic fallback. See IMPROVEMENTS.md for the bug trail.
    is_upgrade = "upgrade" in prompt or "smith" in prompt
    is_transform = "transform" in prompt
    is_discard = "discard" in prompt and "discard pile" not in prompt

    floor = _floor(state)
    run = state.get("run") or {}
    hp = run.get("current_hp", 50)
    max_hp = run.get("max_hp", 80)
    owned_relics = _get_relics(state)

    deck_objs = _build_deck_card_objects(state)

    if is_discard:
        # Discard: drop least valuable card. Resolution order
        # (lowest score wins):
        #   1. Real junk (Card.is_junk) — Wound, Slimed, Clumsy, etc.
        #   2. Carry cargo (Card.is_carry_cargo) — Spoils Map, Lantern
        #      Key, Byrdonis Egg. Inert in combat, so discarding them
        #      loses nothing — preferable to discarding any playable
        #      real card.
        #   3. Unplayable-this-turn (cost < 0 or unplayable_reason set).
        #   4. Lowest-value real card via _organic_removal_score.
        # Uses Card.is_junk/is_carry_cargo via DB lookup — needed because
        # the live state dict has no 'type' field.
        best_idx, best_score, best_name = None, 999.0, ""
        for card in cards:
            name = card.get("name", card.get("card_id", "?"))
            idx = card.get("index", 0)

            if name in protect_cards or name in _LIVE_PROTECTED_CARDS:
                score = 100.0  # never discard protected
            else:
                card_obj = _resolve_card_obj(name)
                cost = card.get("cost", card.get("energy_cost", 0))
                if not isinstance(cost, (int, float)):
                    cost = 0

                if card_obj is not None and card_obj.is_junk:
                    score = -2.0  # real junk — discard first
                elif card_obj is not None and card_obj.is_carry_cargo:
                    score = -1.0  # dead weight — beats any playable card
                elif (card.get("unplayable_reason")
                      or card.get("unplayable")
                      or card.get("is_unplayable")
                      or cost < 0):
                    score = -0.5  # unplayable this turn (but not junk)
                elif card_obj and deck_objs:
                    score = _organic_removal_score(card_obj, deck_objs)
                else:
                    # No deck context — rough fallback on starter cards.
                    base = name.rstrip("+")
                    score = 0.05 if base in ("Strike", "Defend") else 0.40

            if score < best_score:
                best_idx = idx
                best_score = score
                best_name = name
        if best_idx is not None:
            return Decision("select_deck_card", best_idx,
                            f"Discard {best_name} (value={best_score:.2f})")

    if is_remove or is_transform:
        # Remove/transform: organic removal scorer — lower score wins.
        # Protected cards are never touched. Carry-cargo quest cards
        # (Spoils Map, Lantern Key, Byrdonis Egg) are ALSO never removed
        # — they're only "free to discard" in combat-hand prompts; on a
        # permanent remove/transform surface, removing them loses the
        # quest item forever, which is strictly bad.
        best_idx, best_score, best_name = None, 999.0, ""
        for card in cards:
            name = card.get("name", card.get("card_id", "?"))
            idx = card.get("index", 0)
            if name in protect_cards or name in _LIVE_PROTECTED_CARDS:
                continue

            card_obj = _resolve_card_obj(name)
            if card_obj is not None and card_obj.is_carry_cargo:
                continue  # never permanently remove a quest carry card

            if card_obj and deck_objs:
                score = _organic_removal_score(card_obj, deck_objs)
            else:
                base = name.rstrip("+")
                # Weak fallback: starters are best removal targets.
                score = 0.05 if base in ("Strike", "Defend") else 0.40

            if score < best_score:
                best_idx = idx
                best_score = score
                best_name = name

        if best_idx is not None:
            action = "remove" if is_remove else "transform"
            return Decision("select_deck_card", best_idx,
                            f"{action} {best_name} (value={best_score:.2f})")

    elif is_upgrade:
        # Upgrade: highest organic-value target given the current deck.
        best_idx, best_score, best_name = None, -999.0, ""
        for card in cards:
            name = card.get("name", card.get("card_id", "?"))
            idx = card.get("index", 0)

            card_obj = _resolve_card_obj(name)
            if card_obj and deck_objs:
                score = _organic_upgrade_value(
                    card_obj, deck_objs, floor, hp, max_hp, relics=owned_relics)
                # Powers are high-priority upgrades (permanent effects).
                card_type = (card.get("type") or "").lower()
                if card_type == "power":
                    score += 0.15
            else:
                # Fallback: powers > everything else.
                card_type = (card.get("type") or "").lower()
                score = 0.5 if card_type == "power" else 0.2

            if score > best_score:
                best_idx = idx
                best_score = score
                best_name = name

        if best_idx is not None:
            return Decision("select_deck_card", best_idx,
                            f"Upgrade {best_name} (value={best_score:.2f})")

    # Generic selection: highest organic value in the current deck.
    best_idx, best_score, best_name = None, -999.0, ""
    for card in cards:
        name = card.get("name", card.get("card_id", "?"))
        idx = card.get("index", 0)
        card_obj = _resolve_card_obj(name)
        if card_obj and deck_objs:
            try:
                from .card_picker import score_card
                score = score_card(card_obj, deck_objs, floor, hp, max_hp,
                                   relics=owned_relics)
            except Exception:
                score = 0.0
        else:
            score = 0.0
        if score > best_score:
            best_idx = idx
            best_score = score
            best_name = name

    return Decision("select_deck_card", best_idx or 0,
                    f"Selected {best_name} (value={best_score:.2f})")
