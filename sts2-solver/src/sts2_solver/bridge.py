"""Bridge between MCP/HTTP game state and the combat simulator.

Converts raw game state JSON (from the STS2 Agent mod HTTP API) into
the simulator's CombatState, and converts solver Actions back into
MCP-compatible action parameters.

# Non-combat screens (IMPROVEMENTS.md #8)
#
# ``state_from_mcp()`` used to return garbage for rest/map/shop/event
# screens because it read ``combat.*`` fields that were empty. It now
# auto-detects the screen and returns a *combat-less* ``CombatState``:
# empty enemies list, a player with full deck loaded into ``draw_pile``,
# and floor/gold/relics populated from the run data. This is enough for
# the network's option-head to encode deck/relic/HP context — exactly
# what ``_az_decide_*`` in runner.py used to rebuild manually.
#
# Three companion extractors — ``map_options_from_mcp``,
# ``rest_options_from_mcp``, ``shop_options_from_mcp`` — return the
# ``(opt_types, opt_cards, actions)`` tuples the option-head network
# expects, plus the back-map to the MCP action parameters. Training
# code (self_play/full_run) still builds its own options from the
# sim-side data structures, but the two code paths now agree on the
# option-type constants and ordering so the network sees consistent
# encoding across train-time and test-time.
"""

from __future__ import annotations

from .actions import Action
from .data_loader import CardDB
from .enemy_predict import annotate_predictions
from .models import Card, CombatState, EnemyState, PlayerState


_warned_potions: set[str] = set()
_warned_multihit: set[str] = set()


def _parse_potions(raw_potions: list) -> list[dict]:
    """Parse potions from raw game state.

    Classifies potions by keyword matching and returns a list of dicts.
    Unknown potions are logged and included as generic entries (with just 'name').
    Empty slots return empty dicts.
    """
    potions: list[dict] = []
    for idx, p in enumerate(raw_potions):
        if not p.get("occupied"):
            potions.append({})
            continue
        name = (p.get("name") or "").lower()
        pot: dict = {"name": p.get("name", "?")}
        # Classify by keywords (matches simulator POTION_TYPES)
        if any(k in name for k in ("blood", "heal", "fairy", "fruit", "regen")):
            pot["heal"] = 20
        elif any(k in name for k in ("block", "ghost", "shield", "iron", "armor")):
            pot["block"] = 12
        elif "strength" in name or "flex" in name:
            pot["strength"] = 2
        elif any(k in name for k in ("fire", "explosive", "attack")):
            pot["damage_all"] = 20
        elif "weak" in name or "fear" in name:
            pot["enemy_weak"] = 3
        elif any(k in name for k in ("poison", "venom", "toxic", "blight", "corrosive", "orobic", "acid")):
            pot["damage_all"] = 15  # Poison/acid potions — treat as damage
        elif any(k in name for k in ("dexterity", "dex", "agile", "agility")):
            pot["dexterity"] = 2
        elif any(k in name for k in ("draw", "speed", "swift", "skill", "colorless")):
            pot["draw"] = 2  # Card-draw potions
        elif any(k in name for k in ("energy", "bottle", "entropic")):
            pot["energy"] = 1
        elif any(k in name for k in ("blessing", "forge", "smith", "anvil")):
            pot["strength"] = 1  # Blessing-type — modest buff
        elif any(k in name for k in ("rock", "shaped", "potion-shaped", "thrown", "bomb")):
            pot["damage_all"] = 15  # Throwable damage potions
        elif "essence" in name or "elixir" in name:
            pot["heal"] = 15
        elif "smoke" in name or "vanish" in name or "stealth" in name:
            pot["block"] = 10  # Evasion-type
        else:
            # Unknown potion — log ONCE and treat as generic (occupies slot)
            pname = p.get("name", "?")
            if pname not in _warned_potions:
                _warned_potions.add(pname)
                print(f"[bridge] Unknown potion: '{pname}' at slot {idx} — treating as generic")
            potions.append(pot)  # Add generic entry so MCTS knows slot is occupied
            continue
        potions.append(pot)
    return potions


def state_from_mcp(raw: dict, card_db: CardDB,
                   move_indices: dict[tuple[int, str], int] | None = None) -> CombatState:
    """Convert an MCP game state dict into a CombatState.

    Works for both combat and non-combat screens. In combat, returns
    the normal hand/enemies populated state. Out of combat (rest, map,
    shop, event, card reward), returns a *combat-less* state: empty
    enemies list, hand empty, full deck loaded into ``draw_pile``.
    This is the minimum context the network's state encoder and option
    head need to score map/rest/shop/card options.

    Args:
        raw: The full game state dict from GET /state or get_game_state().
        card_db: Card database for resolving card definitions.

    Returns:
        A CombatState ready for the solver or option-head network.
    """
    combat = raw.get("combat") or {}
    player_raw = combat.get("player") or {}
    enemies_raw = combat.get("enemies") or []
    hand_raw = combat.get("hand") or []

    # Non-combat detection: if there's no combat block, or the combat
    # block has no live enemies and no hand, the caller is on a
    # rest/map/shop/event/card_reward screen. Delegate to the non-combat
    # builder so we still return a sensible state.
    live_enemies = [e for e in enemies_raw if e.get("is_alive", True)
                    and e.get("current_hp", 0) > 0]
    if not live_enemies and not hand_raw:
        return _noncombat_state_from_mcp(raw, card_db)

    # Build player state
    player = PlayerState(
        hp=player_raw.get("current_hp", 0),
        max_hp=player_raw.get("max_hp", 0),
        block=player_raw.get("block", 0),
        energy=player_raw.get("energy", 0),
        max_energy=raw.get("run", {}).get("max_energy", 3),
        powers=_parse_powers(player_raw.get("powers", [])),
    )

    # Build hand from runtime card data (has computed values)
    player.hand = [_card_from_runtime(c, card_db) for c in hand_raw]

    # Build draw/discard/exhaust piles from combat state.
    # The API may provide these under combat directly or under combat.player.
    # Parsing them gives the network visibility into deck composition
    # for draw-probability decisions and multi-card planning.
    draw_raw = (combat.get("draw_pile")
                or player_raw.get("draw_pile")
                or [])
    discard_raw = (combat.get("discard_pile")
                   or player_raw.get("discard_pile")
                   or [])
    exhaust_raw = (combat.get("exhaust_pile")
                   or player_raw.get("exhaust_pile")
                   or [])
    player.draw_pile = [_card_from_runtime(c, card_db) for c in draw_raw]
    player.discard_pile = [_card_from_runtime(c, card_db) for c in discard_raw]
    player.exhaust_pile = [_card_from_runtime(c, card_db) for c in exhaust_raw]

    # Build enemies
    enemies = [_enemy_from_runtime(e) for e in enemies_raw]

    # Extract relic IDs for evaluator awareness
    run = raw.get("run") or {}
    relics_raw = run.get("relics") or raw.get("relics") or []
    relic_ids = frozenset(
        r.get("relic_id", r.get("id", "")) if isinstance(r, dict) else str(r)
        for r in relics_raw
    )

    floor = run.get("floor", 0)

    # Predict future enemy intents from move tables
    annotate_predictions(enemies, turns=2, move_indices=move_indices)

    # Parse potions from run state
    potions_raw = run.get("potions") or []
    player.potions = _parse_potions(potions_raw)

    gold = run.get("gold", 0)

    return CombatState(
        player=player,
        enemies=enemies,
        turn=raw.get("turn", 1),
        relics=relic_ids,
        floor=floor,
        gold=gold,
    )


def _noncombat_state_from_mcp(raw: dict, card_db: CardDB) -> CombatState:
    """Build a CombatState for non-combat screens (rest/map/shop/event).

    No enemies, no hand. The deck is parsed from run data and loaded
    into ``draw_pile`` so the network can see card composition. Floor,
    gold, relics, and HP/max_HP are populated normally.

    The returned state is NOT safe to pass to the combat simulator or
    MCTS — it has no enemies. It IS safe to pass to:
    - ``alphazero.encoding.encode_state()`` (reads deck / relics / HP)
    - ``network.encode_state()`` (produces a hidden vector)
    - ``network.pick_best_option()`` (for the option-head network)
    """
    run = raw.get("run") or {}
    relics_raw = run.get("relics") or raw.get("relics") or []
    relic_ids = frozenset(
        r.get("relic_id", r.get("id", "")) if isinstance(r, dict) else str(r)
        for r in relics_raw
    )

    # Parse deck into Card objects using the same logic the live runner
    # used in _az_run_state_tensors before #8. This centralizes it here
    # so every caller gets the same deck parse.
    deck_raw = run.get("deck", [])
    deck_cards: list[Card] = []
    for rc in deck_raw:
        cid = rc.get("card_id") or rc.get("id", "")
        upgraded = bool(rc.get("upgraded"))
        c = card_db.get(cid, upgraded=upgraded)
        if c is None and upgraded:
            c = card_db.get(cid.rstrip("+") + "+")
        if c is None:
            c = card_db.get(cid)
        if c is not None:
            deck_cards.append(c)

    hp = run.get("current_hp", 70)
    max_hp = run.get("max_hp", 70)
    max_energy = run.get("max_energy", 3)
    gold = run.get("gold", 0)
    floor = run.get("floor", 0)

    player = PlayerState(
        hp=hp, max_hp=max_hp, block=0,
        energy=max_energy, max_energy=max_energy,
        powers={},
    )
    player.hand = []
    player.draw_pile = deck_cards
    player.discard_pile = []
    player.exhaust_pile = []
    # Parse potions (same logic as combat path)
    potions_raw = run.get("potions") or []
    player.potions = _parse_potions(potions_raw)

    return CombatState(
        player=player,
        enemies=[],
        turn=0,
        relics=relic_ids,
        floor=floor,
        gold=gold,
    )


# ---------------------------------------------------------------------------
# Per-screen option extractors (IMPROVEMENTS.md #8)
#
# Each extractor returns a ``ScreenOptions`` dict with:
#   opt_types:   list[int]    — option-type constants (OPTION_REST, ...)
#   opt_cards:   list[int]    — vocab indices (0 for no card)
#   actions:     list[tuple]  — (mcp_action_name, mcp_option_index, label)
#
# The three lists are parallel: opt_types[i], opt_cards[i], actions[i]
# all describe the i-th option the option-head network will score.
#
# The training loop in self_play/full_run.py builds its own option
# lists from sim-side data structures; these extractors give the live
# runner.py an equivalent parse that works on raw MCP dicts.
# ---------------------------------------------------------------------------

def map_options_from_mcp(raw: dict) -> dict:
    """Extract map-node options from an MCP map screen.

    Returns dict with ``opt_types``, ``opt_cards``, ``actions``.
    Each action is ``("choose_map_node", game_node_idx, label)``.
    """
    from .alphazero.self_play import ROOM_TYPE_TO_OPTION

    map_data = raw.get("map") or (raw.get("agent_view") or {}).get("map") or {}
    nodes = map_data.get("available_nodes") or map_data.get("nodes") or []

    opt_types: list[int] = []
    opt_cards: list[int] = []
    actions: list[tuple[str, int | None, str]] = []

    for i, node in enumerate(nodes):
        idx = node.get("index", i)
        t = (node.get("node_type") or node.get("type") or
             node.get("icon") or node.get("symbol", "")).lower()

        if "elite" in t:
            rt = "elite"
        elif "rest" in t:
            rt = "rest"
        elif "shop" in t or "merchant" in t:
            rt = "shop"
        elif "event" in t or "unknown" in t or "mystery" in t:
            rt = "event"
        else:
            rt = "normal"  # monster, treasure, etc.

        opt_type = ROOM_TYPE_TO_OPTION.get(rt)
        if opt_type is None:
            continue
        opt_types.append(opt_type)
        opt_cards.append(0)
        actions.append(("choose_map_node", idx, f"{rt} (node {idx})"))

    return {"opt_types": opt_types, "opt_cards": opt_cards, "actions": actions}


def rest_options_from_mcp(raw: dict, deck_cards: list[Card], vocabs) -> dict:
    """Extract rest-site options (rest + smithable card candidates).

    ``deck_cards`` is the parsed deck from ``_noncombat_state_from_mcp``.
    ``vocabs`` is the alphazero ``Vocabularies`` bundle used to look up
    card-name vocab ids.

    Returns a dict with ``opt_types``, ``opt_cards``, and ``actions``:
        actions[0] = ("choose_rest_option", game_rest_idx, "rest")
        actions[1:] = ("choose_rest_option", game_upgrade_idx, "Smith <name>")

    The game-level rest/upgrade indices are parsed from the raw MCP
    selection, falling back to (0, 1) when unavailable.
    """
    from .alphazero.self_play import OPTION_REST, OPTION_SMITH

    # Parse game-level rest/upgrade option indices from the selection.
    selection = raw.get("selection") or {}
    options = selection.get("options") or []
    game_rest_idx: int | None = None
    game_upgrade_idx: int | None = None
    for i, o in enumerate(options):
        label = (o.get("label") or o.get("text") or "").lower()
        if "rest" in label or "heal" in label:
            game_rest_idx = o.get("index", i)
        elif "smith" in label or "upgrade" in label:
            game_upgrade_idx = o.get("index", i)
    if game_rest_idx is None:
        game_rest_idx = 0
    if game_upgrade_idx is None:
        game_upgrade_idx = 1

    opt_types: list[int] = [OPTION_REST]
    opt_cards: list[int] = [0]
    actions: list[tuple[str, int | None, str]] = [
        ("choose_rest_option", game_rest_idx, "rest"),
    ]

    # Smith candidates: every non-upgraded deck card is eligible.
    for card in deck_cards:
        if getattr(card, "upgraded", False):
            continue
        cname = getattr(card, "name", "?")
        cid = getattr(card, "id", "")
        opt_types.append(OPTION_SMITH)
        opt_cards.append(vocabs.cards.get(cid.rstrip("+")) if vocabs else 0)
        actions.append(("choose_rest_option", game_upgrade_idx,
                        f"Smith {cname}"))

    return {"opt_types": opt_types, "opt_cards": opt_cards, "actions": actions}


def shop_options_from_mcp(raw: dict, deck_cards: list[Card], gold: int,
                          vocabs) -> dict:
    """Extract shop options (remove + buy-relic + buy-card + buy-potion + leave).

    Reuses the *same* parse the live runner used inline. Centralizing
    it here means self-play + live play see equivalent option ordering.
    """
    from .alphazero.self_play import (
        OPTION_SHOP_BUY, OPTION_SHOP_BUY_POTION,
        OPTION_SHOP_LEAVE, OPTION_SHOP_REMOVE,
        OPTION_SHOP_BUY_RELIC,
    )

    actions_raw = raw.get("available_actions") or []
    shop = raw.get("shop") or (raw.get("agent_view") or {}).get("shop") or {}

    opt_types: list[int] = []
    opt_cards: list[int] = []
    actions: list[tuple[str, int | None, str]] = []

    # Remove: only Strike/Defend basic cards are worth removing in the
    # shop heuristic path. The option head scores each candidate
    # individually so the best removal target wins.
    if "remove_card_at_shop" in actions_raw:
        remove_cost = shop.get("remove_cost", 75)
        if isinstance(remove_cost, int) and remove_cost <= gold:
            for card in deck_cards:
                if getattr(card, "name", "") in ("Strike", "Defend") and not getattr(card, "upgraded", False):
                    opt_types.append(OPTION_SHOP_REMOVE)
                    cid = getattr(card, "id", "")
                    opt_cards.append(vocabs.cards.get(cid.rstrip("+"))
                                     if vocabs else 0)
                    actions.append(("remove_card_at_shop", None,
                                    f"Remove {card.name}"))

    # Buy relic — highest-impact shop decision, enumerated before cards
    # to match training's option ordering (docs/shop_parity.md #18).
    if "buy_relic" in actions_raw:
        for i, relic_info in enumerate(shop.get("relics", []) or []):
            price = relic_info.get("price", relic_info.get("cost", 999))
            if not isinstance(price, int) or price > gold:
                continue
            rname = (relic_info.get("name") or relic_info.get("relic_id")
                     or relic_info.get("id") or "relic")
            opt_types.append(OPTION_SHOP_BUY_RELIC)
            opt_cards.append(
                vocabs.relics.get(rname) if vocabs else 0)
            actions.append(("buy_relic", i, f"Buy {rname} ({price}g)"))

    # Buy card
    if "buy_card" in actions_raw:
        for i, card_info in enumerate(shop.get("cards", []) or []):
            price = card_info.get("price", card_info.get("cost", 999))
            if not isinstance(price, int) or price > gold:
                continue
            cid = card_info.get("card_id") or card_info.get("id", "")
            opt_types.append(OPTION_SHOP_BUY)
            opt_cards.append(vocabs.cards.get(cid.rstrip("+")) if vocabs else 0)
            name = card_info.get("name", cid)
            actions.append(("buy_card", i, f"Buy {name} ({price}g)"))

    # Buy potion
    if "buy_potion" in actions_raw:
        for i, pot_info in enumerate(shop.get("potions", []) or []):
            price = pot_info.get("price", pot_info.get("cost", 999))
            if not isinstance(price, int) or price > gold:
                continue
            opt_types.append(OPTION_SHOP_BUY_POTION)
            opt_cards.append(0)  # potions aren't cards
            name = pot_info.get("name", "potion")
            actions.append(("buy_potion", i, f"Buy {name} ({price}g)"))

    # Leave is always the fallback
    if "close_shop_inventory" in actions_raw:
        opt_types.append(OPTION_SHOP_LEAVE)
        opt_cards.append(0)
        actions.append(("close_shop_inventory", None, "Leave shop"))

    return {"opt_types": opt_types, "opt_cards": opt_cards, "actions": actions}


def card_reward_options_from_mcp(raw: dict, vocabs) -> dict:
    """Extract card-reward options (take-card-N + skip).

    Mirrors ``full_run._network_pick_card``'s training input shape:
        opt_types = [OPTION_CARD_REWARD] * N + [OPTION_CARD_SKIP]
        opt_cards = [vocab.cards[base_id], ..., 0]
        actions   = [("choose_reward_card", game_idx, "Take <name>"), ...,
                     ("skip_reward_cards", None, "Skip")]

    The live game's per-card ``index`` (as reported by the MCP reward
    payload) is preserved so the runner can pass it straight to
    ``act(choose_reward_card, option_index=...)``.

    Returns ``{}`` (all lists empty) if no card reward payload is
    visible — the caller should fall through to the deterministic path
    in that case.
    """
    from .alphazero.self_play import OPTION_CARD_REWARD, OPTION_CARD_SKIP

    # Extract card options from the several places MCP may put them.
    reward = raw.get("reward") or {}
    cards = (reward.get("card_options") or reward.get("card_choices")
             or reward.get("cards") or [])
    if not cards:
        sel = raw.get("selection") or {}
        cards = sel.get("cards") or []
    if not cards:
        av_reward = (raw.get("agent_view") or {}).get("reward") or {}
        cards = (av_reward.get("card_options") or av_reward.get("card_choices")
                 or av_reward.get("cards") or [])

    opt_types: list[int] = []
    opt_cards: list[int] = []
    actions: list[tuple[str, int | None, str]] = []

    actions_raw = raw.get("available_actions") or []
    take_action = ("choose_reward_card" if "choose_reward_card" in actions_raw
                   else "choose_reward_card")

    for i, card_info in enumerate(cards):
        cid = (card_info.get("card_id") or card_info.get("id")
               or card_info.get("name") or "")
        base_id = cid.rstrip("+")
        name = card_info.get("name") or cid or f"card{i}"
        game_idx = card_info.get("index", i)
        opt_types.append(OPTION_CARD_REWARD)
        opt_cards.append(vocabs.cards.get(base_id) if vocabs else 0)
        actions.append((take_action, game_idx, f"Take {name}"))

    # Skip is always an option (even if cards list is empty) as long as
    # the game permits it. If neither take nor skip is available the
    # caller should fall through to the deterministic handler.
    if "skip_reward_cards" in actions_raw or cards:
        opt_types.append(OPTION_CARD_SKIP)
        opt_cards.append(0)
        actions.append(("skip_reward_cards", None, "Skip"))

    return {"opt_types": opt_types, "opt_cards": opt_cards, "actions": actions}


def event_options_from_mcp(raw: dict, vocabs) -> dict:
    """Extract non-Neow event-choice options in training's encoding.

    V10: uses real ``EVENT_CHOICE_VOCAB`` IDs via a dedicated
    ``event_choice_embed`` table in the network, replacing the old
    ordinal-position placeholder that abused the card embedding.

    Locked options are filtered out, but each surviving option's
    original ``index`` field is preserved so the runner calls
    ``act(choose_event_option, option_index=<game_idx>)`` with exactly
    the same integer the MCP layer expects.

    Returns ``{opt_types, opt_cards, actions}`` where each action is
    ``("choose_event_option", game_idx, label)``. Returns empty lists
    if the event screen is not visible.
    """
    from .alphazero.self_play import OPTION_EVENT_CHOICE
    from .simulator import _event_choice_vocab_id

    event = raw.get("event") or (raw.get("agent_view") or {}).get("event") or {}
    event_id = (event.get("event_id") or event.get("id") or "")
    raw_options = event.get("options") or []
    usable: list[tuple[int, int, dict]] = []  # (canon_i, game_idx, opt)
    for i, opt in enumerate(raw_options):
        if opt.get("locked"):
            continue
        game_idx = opt.get("index", i)
        usable.append((i, game_idx, opt))

    if not usable:
        return {"opt_types": [], "opt_cards": [], "actions": []}

    opt_types: list[int] = []
    opt_cards: list[int] = []
    actions: list[tuple[str, int | None, str]] = []
    for canon_i, game_idx, opt in usable:
        opt_types.append(OPTION_EVENT_CHOICE)
        # V10: real vocab ID from the pre-populated EVENT_CHOICE_VOCAB.
        # Falls back to 0 (UNK) for events not in events.json.
        opt_cards.append(
            _event_choice_vocab_id(event_id, canon_i) if event_id else 0)
        label = (opt.get("name") or opt.get("title")
                 or (opt.get("description") or "")[:40]
                 or f"option {game_idx}")
        actions.append(("choose_event_option", int(game_idx), label))

    return {"opt_types": opt_types, "opt_cards": opt_cards, "actions": actions}


def action_to_mcp(action: Action) -> dict:
    """Convert a solver Action to MCP act() parameters.

    Returns:
        Dict with 'action' and relevant indices.
    """
    if action.action_type == "end_turn":
        return {"action": "end_turn"}

    if action.action_type == "use_potion":
        result = {"action": "use_potion", "option_index": action.potion_idx}
        if action.target_idx is not None:
            result["target_index"] = action.target_idx
        return result

    result = {
        "action": "play_card",
        "card_index": action.card_idx,
    }
    if action.target_idx is not None:
        result["target_index"] = action.target_idx
    else:
        # Game API requires target_index for all cards including AoE.
        # Default to enemy 0 when the solver doesn't specify a target.
        result["target_index"] = 0
    return result


def actions_to_mcp_sequence(actions: list[Action]) -> list[dict]:
    """Convert a list of solver Actions to MCP action dicts."""
    return [action_to_mcp(a) for a in actions]


# ---------------------------------------------------------------------------
# Internal parsing helpers
# ---------------------------------------------------------------------------

def _parse_powers(powers_raw: list[dict]) -> dict[str, int]:
    """Parse runtime powers list into {power_name: amount} dict."""
    result: dict[str, int] = {}
    for p in powers_raw:
        # Runtime format: {"power_id": "VULNERABLE_POWER", "name": "Vulnerable", "amount": 2}
        name = p.get("name", "")
        amount = p.get("amount", 0)
        if name and amount != 0:
            result[name] = amount
    return result


def _card_from_runtime(raw: dict, card_db: CardDB) -> Card:
    """Build a Card from runtime combat hand data.

    Uses the card_db definition for damage/block (base, un-modified values).
    The simulator will apply power modifiers (Strength, Dexterity, Weak, etc.)
    during MCTS search, so we pass BASE values here to avoid double-application
    of power modifiers (the game's runtime values already have them applied).
    """
    card_id = raw.get("card_id", "")
    upgraded = raw.get("upgraded", False)

    # Try to get from card_db
    card = card_db.get(card_id, upgraded=upgraded)

    if card is not None:
        # Use the card_db's base damage/block values (not power-modified).
        # The simulator will apply power modifiers during MCTS.
        # Only override cost with runtime cost (can be modified by Liquid Memories, etc.)
        return Card(
            id=card.id,
            name=card.name,
            cost=raw.get("energy_cost", card.cost),
            card_type=card.card_type,
            target=card.target,
            upgraded=upgraded,
            damage=card.damage,
            block=card.block,
            hit_count=card.hit_count,
            powers_applied=card.powers_applied,
            cards_draw=card.cards_draw,
            energy_gain=card.energy_gain,
            hp_loss=card.hp_loss,
            keywords=card.keywords,
            tags=card.tags,
            spawns_cards=card.spawns_cards,
            is_x_cost=card.is_x_cost,
        )

    # Card not in DB - build a minimal Card from runtime data.
    # NOTE: For unknown cards, we must use runtime values, which may already
    # include power modifications. The simulator will apply them again, potentially
    # causing double-application. This should be rare (only for mod cards).
    from .constants import CardType, TargetType

    dynamic = {dv["name"]: dv["base_value"] for dv in raw.get("dynamic_values", [])}

    target_str = raw.get("target_type", "Self")
    try:
        target = TargetType(target_str)
    except ValueError:
        print(f"[bridge] Unknown target_type '{target_str}' for card {card_id}, defaulting to SELF")
        target = TargetType.SELF

    card_type_str = raw.get("card_type", "Skill")
    try:
        card_type = CardType(card_type_str)
    except ValueError:
        card_type = CardType.SKILL

    # Check for Unplayable keyword or -1 cost (Status/Curse cards)
    runtime_cost = raw.get("energy_cost", 0)
    runtime_keywords = raw.get("keywords") or []
    is_unplayable = runtime_cost == -1 or any(
        (k.lower() if isinstance(k, str) else "") == "unplayable"
        for k in runtime_keywords
    )

    return Card(
        id=card_id,
        name=raw.get("name", card_id),
        cost=-1 if is_unplayable else runtime_cost,
        card_type=card_type,
        target=target,
        upgraded=upgraded,
        damage=dynamic.get("Damage"),
        block=dynamic.get("Block"),
    )


def _enemy_from_runtime(raw: dict) -> EnemyState:
    """Build an EnemyState from runtime combat enemy data."""
    # Parse intents — try multiple possible API field names
    intents = raw.get("intents") or raw.get("intent") or []
    if isinstance(intents, dict):
        intents = [intents]  # single intent as dict → wrap in list

    # Also try move/next_move fields for alternative API formats
    if not intents:
        move = raw.get("move") or raw.get("next_move") or {}
        if move:
            intents = [move] if isinstance(move, dict) else move

    intent_type = None
    intent_damage = None
    intent_hits = 1
    intent_block = None
    intent_effects: dict = {}  # capture buff/debuff details

    for intent in intents:
        if not isinstance(intent, dict):
            continue
        it = (intent.get("intent_type") or intent.get("type") or "").lower()

        if "attack" in it:
            intent_type = "Attack"
            intent_damage = intent.get("damage") or intent.get("base_damage")
            intent_hits = intent.get("hits") or intent.get("times") or 1
        if "defend" in it or "block" in it:
            if intent_type is None:
                intent_type = "Defend"
            intent_block = intent.get("block") or intent.get("amount")
        if "buff" in it:
            intent_type = intent_type or "Buff"
            intent_effects["buff"] = True
        if "debuff" in it:
            intent_type = intent_type or "Debuff"
            intent_effects["debuff"] = True
        if "status" in it:
            intent_type = intent_type or "StatusCard"

    # Validate intent_type against known values
    known_intent_types = {"Attack", "Defend", "Buff", "Debuff", "StatusCard", "Unknown"}
    if intent_type is not None and intent_type not in known_intent_types:
        print(f"[bridge] Unknown intent_type '{intent_type}' for enemy {raw.get('name', 'UNKNOWN')}, raw data: {intents}")

    # Log multi-hit intents for verification (once per enemy name)
    if intent_hits > 1:
        enemy_key = raw.get("name", "UNKNOWN")
        if enemy_key not in _warned_multihit:
            _warned_multihit.add(enemy_key)
            print(f"[bridge] Multi-hit intent: {enemy_key} - {intent_type} x{intent_hits} (damage={intent_damage}), raw: {intents}")

    return EnemyState(
        id=raw.get("enemy_id") or raw.get("id", ""),
        name=raw.get("name", ""),
        hp=raw.get("current_hp", 0),
        max_hp=raw.get("max_hp", 0),
        block=raw.get("block", 0),
        powers=_parse_powers(raw.get("powers", [])),
        intent_type=intent_type,
        intent_damage=intent_damage,
        intent_hits=intent_hits,
        intent_block=intent_block,
    )
