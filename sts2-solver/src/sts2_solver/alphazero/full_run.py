"""Full Act 1 run training for AlphaZero.

Plays complete runs using MCTS for combat + deterministic advisor for
non-combat decisions. The network learns HP conservation across combats
and plays with naturally evolving decks.

Value targets: based on floor reached + HP remaining, giving continuous
signal across the full run. Early combats where the player took too
much damage get lower values because the run died later.
"""

from __future__ import annotations

import math
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import torch

from ..actions import Action, END_TURN, enumerate_actions
from ..combat_engine import (
    can_play_card,
    end_combat_relics,
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_combat,
    start_turn,
    tick_enemy_powers,
)
from ..card_picker import classify_deck
from ..data_loader import CardDB, load_cards
from ..models import Card, CombatState, EnemyState, PlayerState
from ..simulator import (
    _ensure_data_loaded,
    _CHARACTERS_BY_ID,
    _ACTS_BY_ID,
    _ENCOUNTERS_BY_ID,
    _spawn_enemy,
    _create_enemy_ai,
    _set_enemy_intents,
    _resolve_sim_intents,
    _generate_act1_map,
    _generate_act1_map_with_choices,
    _pick_encounter,
    _build_card_pool,
    _offer_card_rewards,
    _pick_card_reward,
    _rest_site_decision,
    _simulate_event,
    _simulate_shop,
    _normalize_card_id,
    GOLD_REWARDS,
    POTION_DROP_CHANCE,
    POTION_SLOTS,
    POTION_TYPES,
    SHOP_CARD_REMOVE_COST,
    SHOP_CARD_COSTS,
    SHOP_POTION_COST,
)

from .encoding import EncoderConfig, Vocabs
from .mcts import MCTS, scale_simulations
from .network import STS2Network
from .self_play import (
    TrainingSample, OptionSample,
    OPTION_REST, OPTION_SMITH, OPTION_SHOP_REMOVE, OPTION_SHOP_BUY,
    OPTION_SHOP_LEAVE, OPTION_CARD_REWARD, OPTION_CARD_SKIP,
    OPTION_SHOP_BUY_POTION, ROOM_TYPE_TO_OPTION,
    _affordable_play_actions,
)
from ..effects import discard_card_from_hand
from .state_tensor import encode_state, encode_actions

# Relics that can drop from elites (effects implemented in combat_engine.py)
ELITE_RELIC_POOL = [
    "ANCHOR", "BLOOD_VIAL", "BAG_OF_PREPARATION", "BRONZE_SCALES",
    "BAG_OF_MARBLES", "FESTIVE_POPPER", "LANTERN", "ODDLY_SMOOTH_STONE",
    "STRIKE_DUMMY", "CLOAK_CLASP", "ART_OF_WAR", "MEAT_ON_THE_BONE",
    "KUNAI", "ORNAMENTAL_FAN", "NUNCHAKU", "LETTER_OPENER", "SHURIKEN",
]

# Character starter relics
STARTER_RELICS = {
    "SILENT": "RING_OF_THE_SNAKE",
    "IRONCLAD": "BURNING_BLOOD",
}


# ---------------------------------------------------------------------------
# MCTS-based combat within a full run
# ---------------------------------------------------------------------------

def mcts_combat(
    deck: list[Card],
    player_hp: int,
    player_max_hp: int,
    player_max_energy: int,
    encounter_id: str,
    card_db: CardDB,
    mcts: MCTS,
    vocabs: Vocabs,
    config: EncoderConfig,
    rng: random.Random,
    mcts_simulations: int = 100,
    temperature: float = 1.0,
    max_turns: int = 30,
    potions: list[dict] | None = None,
    relics: frozenset[str] | None = None,
    is_boss: bool = False,
    detail_log: bool = False,
) -> tuple[list[TrainingSample], str, int, int, list[dict], dict | None]:
    """Run one combat using MCTS.

    Returns (samples, outcome, turns, hp_after, remaining_potions, detail).
    `detail` is a rich per-turn log populated when `detail_log=True`
    (currently used for boss fights so we can analyse play patterns).
    It is None otherwise to keep overhead zero for normal combats.
    """
    _ensure_data_loaded()

    enc = _ENCOUNTERS_BY_ID.get(encounter_id, {})
    monster_list = enc.get("monsters", [])
    enemies: list[EnemyState] = []
    enemy_ais = []
    for m in monster_list:
        mid = m["id"]
        enemies.append(_spawn_enemy(mid))
        enemy_ais.append(_create_enemy_ai(mid))

    if not enemies:
        return [], "win", 0, player_hp, potions or [], None

    draw_pile = list(deck)
    rng.shuffle(draw_pile)
    player = PlayerState(
        hp=player_hp, max_hp=player_max_hp,
        energy=player_max_energy, max_energy=player_max_energy,
        draw_pile=draw_pile,
        potions=[dict(p) for p in (potions or [])],
    )
    state = CombatState(player=player, enemies=enemies, relics=relics or frozenset())
    start_combat(state)

    # --- Initialise detail log (boss fights use this) ---
    detail: dict | None = None
    if detail_log:
        from collections import Counter as _Counter
        _deck_counts = _Counter(c.id for c in deck)
        detail = {
            "encounter_id": encounter_id,
            "is_boss": is_boss,
            "monster_ids": [m.get("id") for m in monster_list],
            "monster_start_hp": [e.hp for e in enemies],
            "monster_max_hp": [e.max_hp for e in enemies],
            "player_start_hp": player_hp,
            "player_max_hp": player_max_hp,
            "player_max_energy": player_max_energy,
            "deck_size": len(deck),
            "deck_counts": dict(_deck_counts),
            "relics": sorted(list(relics)) if relics else [],
            "potions_at_start": [p.get("id") if isinstance(p, dict) else None
                                 for p in (potions or [])],
            "turns": [],
            "outcome": None,
            "final_player_hp": None,
            "total_turns": 0,
            "potions_used": [],
        }

    # Boss fights: dump all offensive potions pre-combat and heal if low.
    # This front-loads burst damage/buffs when it matters most.
    # Non-boss fights: MCTS decides potion usage organically (save for boss).
    if is_boss and state.player.potions:
        from ..simulator import _use_precombat_potions, _use_emergency_potion
        pre_potions = [p for p in state.player.potions if p]
        _potions_before_dump = [p.get("id") for p in pre_potions if p]
        remaining = _use_precombat_potions(state, pre_potions)
        if state.player.hp < state.player.max_hp * 0.40:
            remaining = _use_emergency_potion(state, remaining)
        # Update the player's potion list (keep slots, clear used ones)
        state.player.potions = remaining
        if detail is not None:
            _after_ids = [p.get("id") for p in remaining if p]
            _used = [pid for pid in _potions_before_dump if pid not in _after_ids]
            detail["precombat_potions_used"] = _used
            detail["potions_used"].extend(_used)

    samples: list[TrainingSample] = []
    turn_sample_ranges: list[tuple[int, int]] = []  # (start_idx, end_idx) per turn
    outcome = None

    for turn_num in range(1, max_turns + 1):
        start_turn(state)
        _set_enemy_intents(state, enemy_ais)

        # --- Per-turn detail snapshot ---
        turn_record: dict | None = None
        if detail is not None:
            turn_record = {
                "turn": turn_num,
                "energy_start": state.player.energy,
                "hand_start": [c.id for c in state.player.hand],
                "draw_pile_size": len(state.player.draw_pile),
                "discard_pile_size": len(state.player.discard_pile),
                "hp_start": state.player.hp,
                "enemy_hp_start": [e.hp for e in state.enemies],
                "enemy_intents": [
                    {"type": e.intent_type, "dmg": e.intent_damage, "hits": e.intent_hits}
                    for e in state.enemies
                ],
                "cards_played": [],
                "targets": [],
                "potions_used": [],
                "wasted_end_turn": False,
                "forced_overrides": 0,
            }

        turn_start_sample = len(samples)
        cards_this_turn = 0
        while cards_this_turn < 12:
            outcome = is_combat_over(state)
            if outcome:
                break

            actions = enumerate_actions(state)
            if not actions:
                break

            state_tensors = encode_state(state, vocabs, config)
            action_card_ids, action_features, action_mask = encode_actions(actions, state, vocabs, config)

            scaled_sims = scale_simulations(mcts_simulations, len(actions), is_boss=is_boss)
            action, policy, _root_value = mcts.search(
                state, num_simulations=scaled_sims,
                temperature=temperature,
            )

            # Force-play override: if MCTS wants END_TURN but affordable
            # non-junk cards exist, override 80% of the time to generate
            # exploration data showing the value of playing those cards.
            wasted = False
            if action.action_type == "end_turn":
                affordable = _affordable_play_actions(actions, state)
                if affordable:
                    wasted = True
                    if turn_record is not None:
                        turn_record["wasted_end_turn"] = True
                    if rng.random() < 0.80:
                        action = rng.choice(affordable)
                        if turn_record is not None:
                            turn_record["forced_overrides"] += 1

            samples.append(TrainingSample(
                state_tensors=state_tensors,
                policy=policy,
                value=0.0,  # Filled after run ends
                action_card_ids=action_card_ids,
                action_features=action_features,
                action_mask=action_mask,
                num_actions=len(actions),
                wasted_energy=wasted,
            ))

            if action.action_type == "end_turn":
                break

            if action.action_type == "choose_card":
                # Resolve pending choice (discard, etc.) — doesn't count as a card play
                if action.choice_idx is not None and state.pending_choice is not None:
                    pc = state.pending_choice
                    if pc.choice_type == "discard_from_hand":
                        if action.choice_idx < len(state.player.hand):
                            discard_card_from_hand(state, action.choice_idx)
                        pc.chosen_so_far.append(action.choice_idx)
                        if len(pc.chosen_so_far) >= pc.num_choices:
                            state.pending_choice = None
            elif action.action_type == "use_potion":
                from ..combat_engine import use_potion as _use_potion
                if action.potion_idx is not None:
                    if turn_record is not None:
                        # Capture the potion id before it's consumed
                        try:
                            _pot = state.player.potions[action.potion_idx]
                            if _pot:
                                turn_record["potions_used"].append(_pot.get("id"))
                                detail["potions_used"].append(_pot.get("id"))
                        except Exception:
                            pass
                    _use_potion(state, action.potion_idx)
                cards_this_turn += 1  # count toward safety cap
            elif action.card_idx is not None and can_play_card(state, action.card_idx):
                if turn_record is not None:
                    try:
                        _card = state.player.hand[action.card_idx]
                        turn_record["cards_played"].append(_card.id)
                        turn_record["targets"].append(action.target_idx)
                    except Exception:
                        pass
                play_card(state, action.card_idx, action.target_idx, card_db)
                cards_this_turn += 1

            outcome = is_combat_over(state)
            if outcome:
                break

        turn_end_sample = len(samples)
        turn_sample_ranges.append((turn_start_sample, turn_end_sample))

        # Snapshot post-player-phase (before enemy phase resolves)
        if turn_record is not None:
            turn_record["enemy_hp_after_player"] = [e.hp for e in state.enemies]
            turn_record["damage_dealt"] = sum(
                max(0, before - after)
                for before, after in zip(
                    turn_record["enemy_hp_start"],
                    turn_record["enemy_hp_after_player"],
                )
            )
            turn_record["block_played"] = state.player.block
            # If combat ended during player phase, log and append.
            if outcome:
                turn_record["hp_end"] = state.player.hp
                turn_record["damage_taken"] = 0
                turn_record["enemy_hp_end"] = [e.hp for e in state.enemies]
                turn_record["ended_combat"] = True
                detail["turns"].append(turn_record)
                turn_record = None

        if outcome:
            break

        # -- Intent-aware reward signals --
        # Compute expected incoming damage from enemy intents
        incoming_damage = 0
        for e in state.enemies:
            if e.is_alive and e.intent_type == "Attack" and e.intent_damage:
                incoming_damage += e.intent_damage * max(1, e.intent_hits)

        # Snapshot state before enemy phase
        hp_before_enemy = state.player.hp
        player_block_played = state.player.block  # block accumulated this turn
        hand_block_available = sum(
            c.block for c in state.player.hand
            if c.block and c.block > 0 and c.cost <= state.player.energy
        )
        hand_damage_available = sum(
            c.damage for c in state.player.hand
            if c.damage and c.damage > 0 and c.cost <= state.player.energy
        )

        # Signal 1: Offensive-when-safe — penalise block plays when
        # no enemies intend to attack (wasted tempo).
        if (incoming_damage == 0 and player_block_played > 0
                and turn_start_sample < turn_end_sample):
            # Scale by how much block was wasted relative to energy
            safe_block_penalty = min(0.08, player_block_played / max(1, player_max_hp) * 0.3)
            for idx in range(turn_start_sample, turn_end_sample):
                samples[idx].value_penalty += safe_block_penalty

        # Signal 2: Lethal-awareness — penalise not killing an enemy
        # that could have been killed this turn (they'll deal more
        # damage on future turns if left alive).
        for e in state.enemies:
            if e.is_alive and e.hp > 0 and e.hp <= hand_damage_available:
                # An enemy was killable but survived
                lethal_penalty = min(0.10, e.hp / max(1, player_max_hp) * 0.25)
                for idx in range(turn_start_sample, turn_end_sample):
                    samples[idx].value_penalty += lethal_penalty
                break  # only penalise for one missed lethal per turn

        end_turn(state)
        resolve_enemy_intents(state)
        _resolve_sim_intents(state, enemy_ais)
        tick_enemy_powers(state)

        hp_after_enemy = state.player.hp
        damage_taken = max(0, hp_before_enemy - hp_after_enemy)

        # Finalise this turn's detail record
        if turn_record is not None:
            turn_record["hp_end"] = state.player.hp
            turn_record["damage_taken"] = damage_taken
            turn_record["incoming_damage"] = incoming_damage
            turn_record["enemy_hp_end"] = [e.hp for e in state.enemies]
            detail["turns"].append(turn_record)
            turn_record = None

        # Signal 3: Intent-weighted blocking — penalise based on how
        # far actual block was from optimal (matching incoming damage).
        # Under-blocking penalises harder than over-blocking.
        if turn_start_sample < turn_end_sample and incoming_damage > 0:
            if player_block_played < incoming_damage:
                # Under-blocked: took avoidable damage
                gap = incoming_damage - player_block_played
                under_penalty = min(0.15, gap / max(1, player_max_hp) * 0.4)
                # Only penalise if they had block cards available
                if hand_block_available > 0:
                    for idx in range(turn_start_sample, turn_end_sample):
                        samples[idx].value_penalty += under_penalty
            elif player_block_played > incoming_damage * 1.5:
                # Over-blocked by >50%: wasted resources on defence
                excess = player_block_played - incoming_damage
                over_penalty = min(0.06, excess / max(1, player_max_hp) * 0.15)
                for idx in range(turn_start_sample, turn_end_sample):
                    samples[idx].value_penalty += over_penalty

        outcome = is_combat_over(state)
        if outcome:
            break

    if outcome is None:
        outcome = "lose"

    hp_after = max(0, state.player.hp) if outcome == "win" else 0
    remaining_potions = [p for p in state.player.potions if p]

    # Finalise detail log
    if detail is not None:
        detail["outcome"] = outcome
        detail["final_player_hp"] = hp_after
        detail["total_turns"] = turn_num
        detail["final_enemy_hp"] = [e.hp for e in state.enemies]

    return samples, outcome, turn_num, hp_after, remaining_potions, detail


# ---------------------------------------------------------------------------
# Network-based card reward selection
# ---------------------------------------------------------------------------

def _network_pick_card(
    offered: list[Card],
    deck: list[Card],
    hp: int,
    max_hp: int,
    floor: int,
    mcts: MCTS,
    vocabs: Vocabs,
    config: EncoderConfig,
    card_db: CardDB,
) -> tuple[Card | None, OptionSample | None]:
    """Pick a card reward using the organic card picker.

    The organic picker uses property-based archetype detection, momentum
    scoring, and alpha-blended ML handoff — it's designed to build coherent
    decks from game one. The neural network's option head still trains on
    the card picks (via OptionSample) so it can learn deck-building over
    time, but the actual decision comes from the organic system.

    Returns (picked_card_or_None, training_sample_or_None).
    """
    if not offered:
        return None, None

    # --- Organic picker makes the decision ---
    from ..card_picker import pick_card as organic_pick
    pick = organic_pick(offered, deck, floor, hp, max_hp)

    # --- Build a training sample for the option head (learns from the pick) ---
    sample = None
    try:
        network = mcts.network
        player = PlayerState(
            hp=hp, max_hp=max_hp, energy=3, max_energy=3,
            hand=[], draw_pile=list(deck),
        )
        dummy_state = CombatState(player=player, enemies=[], turn=0, floor=floor)

        state_tensors = encode_state(dummy_state, vocabs, config)
        state_tensors = {k: v.to(mcts.device) for k, v in state_tensors.items()}

        opt_types = [OPTION_CARD_REWARD] * len(offered) + [OPTION_CARD_SKIP]
        opt_cards = []
        for card in offered:
            base_id = card.id.rstrip("+")
            opt_cards.append(vocabs.cards.get(base_id))
        opt_cards.append(0)  # PAD for skip

        # Record which index the organic picker chose
        if pick is None:
            chosen_idx = len(offered)  # skip
        else:
            chosen_idx = next(
                (i for i, c in enumerate(offered) if c.id == pick.id),
                len(offered),
            )

        sample = OptionSample(
            state_tensors={k: v.cpu() for k, v in state_tensors.items()},
            option_types=opt_types,
            option_cards=opt_cards,
            chosen_idx=chosen_idx,
            value=0.0,  # Filled after run ends
        )
    except Exception:
        pass  # Sample building failed — pick still valid

    return pick, sample


# ---------------------------------------------------------------------------
# Full Act 1 run with MCTS combat
# ---------------------------------------------------------------------------

@dataclass
class FullRunResult:
    outcome: str  # "win" or "lose"
    floor_reached: int
    final_hp: int
    max_hp: int
    combats_won: int
    combats_fought: int
    deck_size: int
    samples: list[TrainingSample]
    deck_samples: list  # OptionSample list (card rewards, routed through option head)
    option_samples: list  # OptionSample list (rest/map/shop)
    combat_log: list[dict]
    archetype: str = "undecided"     # emergent archetype of final deck
    archetype_commitment: float = 0.0  # 0.0–1.0
    boss_detail: dict | None = None  # Rich per-turn log of the boss combat
    final_deck: list[str] | None = None  # Card IDs in deck at boss-fight time


def play_full_run(
    mcts: MCTS,
    card_db: CardDB,
    vocabs: Vocabs,
    config: EncoderConfig,
    character: str = "SILENT",
    mcts_simulations: int = 100,
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> FullRunResult:
    """Play a full Act 1 run. Returns result with training samples."""
    if rng is None:
        rng = random.Random()

    _ensure_data_loaded()

    # Character setup
    char_data = _CHARACTERS_BY_ID.get(character, {})
    hp = char_data.get("starting_hp", 70)
    max_hp = hp
    gold = char_data.get("starting_gold", 99)
    max_energy = char_data.get("max_energy", 3)

    # Build starting deck
    raw_deck_ids = char_data.get("starting_deck", [])
    deck: list[Card] = []
    for raw_id in raw_deck_ids:
        card_id = _normalize_card_id(raw_id)
        card = card_db.get(card_id) or card_db.get(raw_id)
        if card:
            deck.append(card)

    if not deck:
        # Fallback: Silent starter
        for name, count in [("STRIKE_SILENT", 5), ("DEFEND_SILENT", 5),
                            ("NEUTRALIZE", 1), ("SURVIVOR", 1)]:
            c = card_db.get(name)
            if c:
                deck.extend([c] * count)

    # Card pools
    char_color = char_data.get("color", "green")
    color_map = {"red": "ironclad", "green": "silent", "blue": "defect",
                 "purple": "necrobinder", "yellow": "regent"}
    card_color = color_map.get(char_color, char_color)
    pools = _build_card_pool(card_db, card_color)

    # Act data + map
    act_data = _ACTS_BY_ID.get("OVERGROWTH", {})
    room_sequence = _generate_act1_map_with_choices(rng)

    # Starter relic
    relics: set[str] = set()
    starter_relic = STARTER_RELICS.get(character)
    if starter_relic:
        relics.add(starter_relic)

    # Run state
    all_samples: list[TrainingSample] = []
    deck_change_samples: list = []
    option_samples: list = []
    combat_samples_by_floor: dict[int, list[TrainingSample]] = {}
    combat_hp_data: dict[int, tuple[int, int, int]] = {}  # floor -> (hp_before, hp_after, potions_used)
    boss_floors: set[int] = set()
    combat_log: list[dict] = []
    combats_won = 0
    combats_fought = 0
    boss_detail_holder: dict | None = None
    potions: list[dict] = []
    seen_encounters: set[str] = set()
    events_list = list(act_data.get("events", []))
    rng.shuffle(events_list)
    event_idx = 0
    floor_reached = 0

    for floor_num, room_entry in enumerate(room_sequence, 1):
        floor_reached = floor_num

        # Resolve map choice nodes via network
        if isinstance(room_entry, list):
            try:
                network = mcts.network
                player = PlayerState(hp=hp, max_hp=max_hp, energy=3, max_energy=3,
                                     draw_pile=list(deck))
                dummy = CombatState(player=player, enemies=[], floor=floor_num, gold=gold, relics=frozenset(relics))
                st = encode_state(dummy, vocabs, config)
                st = {k: v.to(mcts.device) for k, v in st.items()}

                opt_types = [ROOM_TYPE_TO_OPTION[rt] for rt in room_entry]
                opt_cards = [0] * len(room_entry)

                with torch.no_grad():
                    hidden = network.encode_state(**st)
                    best_idx, scores = network.pick_best_option(
                        hidden, opt_types, opt_cards)

                option_samples.append(OptionSample(
                    state_tensors={k: v.cpu() for k, v in st.items()},
                    option_types=opt_types, option_cards=opt_cards,
                    chosen_idx=best_idx, value=0.0,
                ))
                room_type = room_entry[best_idx]
            except Exception:
                room_type = rng.choice(room_entry)
        else:
            room_type = room_entry

        if room_type in ("weak", "normal", "elite", "boss"):
            enc_id = _pick_encounter(act_data, room_type, rng, seen_encounters)
            if enc_id is None:
                continue

            _is_boss = (room_type == "boss")
            potions_before = len([p for p in potions if p])
            samples, outcome, turns, hp_after, potions, combat_detail = mcts_combat(
                deck=deck, player_hp=hp, player_max_hp=max_hp,
                player_max_energy=max_energy, encounter_id=enc_id,
                card_db=card_db, mcts=mcts, vocabs=vocabs, config=config,
                rng=rng, mcts_simulations=mcts_simulations,
                temperature=temperature, potions=potions,
                relics=frozenset(relics),
                is_boss=_is_boss,
                detail_log=_is_boss,  # only log boss combats (overhead)
            )
            if _is_boss and combat_detail is not None:
                boss_detail_holder = combat_detail  # captured into FullRunResult below
            potions_after = len([p for p in potions if p])
            potions_used = max(0, potions_before - potions_after)

            combats_fought += 1
            combat_samples_by_floor[floor_num] = samples
            combat_hp_data[floor_num] = (hp, hp_after, potions_used)
            if room_type == "boss":
                boss_floors.add(floor_num)
            all_samples.extend(samples)

            combat_log.append({
                "floor": floor_num, "encounter": enc_id,
                "room_type": room_type, "outcome": outcome,
                "turns": turns, "hp_before": hp, "hp_after": hp_after,
            })

            if outcome == "lose":
                # Assign values: run died here
                _assign_run_values(combat_samples_by_floor, floor_reached,
                                   len(room_sequence), 0, max_hp,
                                   deck_change_samples, option_samples,
                                   combat_hp_data=combat_hp_data,
                                   boss_floors=boss_floors)
                _arch = classify_deck(deck)
                return FullRunResult(
                    outcome="lose", floor_reached=floor_reached,
                    final_hp=0, max_hp=max_hp,
                    combats_won=combats_won, combats_fought=combats_fought,
                    deck_size=len(deck), samples=all_samples,
                    deck_samples=deck_change_samples,
                    option_samples=option_samples, combat_log=combat_log,
                    archetype=_arch.archetype,
                    archetype_commitment=_arch.commitment,
                    boss_detail=boss_detail_holder,
                    final_deck=[c.id for c in deck],
                )

            combats_won += 1

            # End-of-combat relic effects (healing etc.)
            # Build a temporary state to apply relic effects
            _post_player = PlayerState(hp=hp_after, max_hp=max_hp, energy=0, max_energy=0)
            _post_state = CombatState(player=_post_player, enemies=[], relics=frozenset(relics))
            end_combat_relics(_post_state)
            hp = _post_state.player.hp

            # Post-combat: gold, potions, card/relic rewards
            gold_range = GOLD_REWARDS.get(room_type, (10, 20))
            gold += rng.randint(*gold_range)

            if rng.random() < POTION_DROP_CHANCE and len(potions) < POTION_SLOTS:
                pot = rng.choice(POTION_TYPES)
                potions.append(dict(pot))

            # Elite relic drop
            if room_type == "elite":
                available = [r for r in ELITE_RELIC_POOL if r not in relics]
                if available:
                    relics.add(rng.choice(available))

            if room_type != "boss":
                offered = _offer_card_rewards(pools, deck)
                pick, deck_sample = _network_pick_card(
                    offered, deck, hp, max_hp, floor_num,
                    mcts, vocabs, config, card_db,
                )
                if pick:
                    deck.append(pick)
                if deck_sample:
                    deck_change_samples.append(deck_sample)

            if room_type == "boss":
                _assign_run_values(combat_samples_by_floor, floor_reached,
                                   len(room_sequence), hp, max_hp,
                                   deck_change_samples, option_samples,
                                   combat_hp_data=combat_hp_data,
                                   boss_floors=boss_floors)
                _arch = classify_deck(deck)
                return FullRunResult(
                    outcome="win", floor_reached=floor_reached,
                    final_hp=hp, max_hp=max_hp,
                    combats_won=combats_won, combats_fought=combats_fought,
                    deck_size=len(deck), samples=all_samples,
                    deck_samples=deck_change_samples,
                    option_samples=option_samples, combat_log=combat_log,
                    archetype=_arch.archetype,
                    archetype_commitment=_arch.commitment,
                    boss_detail=boss_detail_holder,
                    final_deck=[c.id for c in deck],
                )

        elif room_type == "rest":
            # Network-scored rest site decision
            try:
                network = mcts.network
                player = PlayerState(hp=hp, max_hp=max_hp, energy=3, max_energy=3,
                                     draw_pile=list(deck))
                dummy = CombatState(player=player, enemies=[], floor=floor_num, gold=gold, relics=frozenset(relics))
                st = encode_state(dummy, vocabs, config)
                st = {k: v.to(mcts.device) for k, v in st.items()}

                opt_types = [OPTION_REST]
                opt_cards = [0]
                deck_indices = [None]  # maps option idx → deck idx

                for di, card in enumerate(deck):
                    if not card.upgraded and card.card_type not in ("Status", "Curse"):
                        up = card_db.get_upgraded(card.id)
                        if up:
                            opt_types.append(OPTION_SMITH)
                            opt_cards.append(vocabs.cards.get(card.id.rstrip("+")))
                            deck_indices.append(di)

                with torch.no_grad():
                    hidden = network.encode_state(**st)
                    best_idx, scores = network.pick_best_option(
                        hidden, opt_types, opt_cards)

                option_samples.append(OptionSample(
                    state_tensors={k: v.cpu() for k, v in st.items()},
                    option_types=opt_types, option_cards=opt_cards,
                    chosen_idx=best_idx, value=0.0,
                ))

                if best_idx == 0:
                    hp = min(hp + int(max_hp * 0.3), max_hp)
                else:
                    di = deck_indices[best_idx]
                    if di is not None and di < len(deck):
                        upgraded = card_db.get_upgraded(deck[di].id)
                        if upgraded:
                            deck[di] = upgraded
            except Exception:
                # Fallback to heuristic
                decision = _rest_site_decision(hp, max_hp, deck, card_db, rng)
                if decision["action"] == "rest":
                    hp = min(hp + decision["hp_delta"], max_hp)
                else:
                    idx = decision["upgrade_card_idx"]
                    if idx is not None and idx < len(deck):
                        upgraded = card_db.get_upgraded(deck[idx].id)
                        if upgraded:
                            deck[idx] = upgraded

        elif room_type == "event":
            if event_idx < len(events_list):
                eid = events_list[event_idx]
                event_idx += 1
            else:
                eid = rng.choice(events_list) if events_list else None
            if eid:
                changes = _simulate_event(eid, deck, hp, max_hp, gold, card_db, rng)
                hp = max(1, min(hp + changes["hp_delta"], max_hp + changes["max_hp_delta"]))
                max_hp += changes["max_hp_delta"]
                gold = max(0, gold + changes["gold_delta"])
                for idx in sorted(changes["cards_removed"], reverse=True):
                    if idx < len(deck):
                        deck.pop(idx)
                for card in changes["cards_added"]:
                    deck.append(card)

        elif room_type == "shop":
            # Network-driven multi-step shop
            try:
                network = mcts.network
                shop_cards = _offer_card_rewards(pools, deck, 3)
                shop_costs = []
                for sc in shop_cards:
                    cost = 75
                    for rarity, pool_cards in pools.items():
                        if any(c.id == sc.id for c in pool_cards):
                            cost = SHOP_CARD_COSTS.get(rarity, 75)
                            break
                    shop_costs.append(cost)

                # Offer 2 random potions at the shop
                shop_potions = [rng.choice(POTION_TYPES) for _ in range(2)]

                for _step in range(6):
                    player = PlayerState(hp=hp, max_hp=max_hp, energy=3,
                                         max_energy=3, draw_pile=list(deck),
                                         potions=[dict(p) for p in potions])
                    dummy = CombatState(player=player, enemies=[],
                                        floor=floor_num, gold=gold)
                    st = encode_state(dummy, vocabs, config)
                    st = {k: v.to(mcts.device) for k, v in st.items()}

                    opt_types = []
                    opt_cards = []
                    actions = []  # ("remove", deck_idx) | ("buy", shop_idx, cost) | ("potion", pot_idx) | ("leave",)

                    # Remove options (Strike/Defend only)
                    if gold >= SHOP_CARD_REMOVE_COST:
                        for di, card in enumerate(deck):
                            if card.name in ("Strike", "Defend") and not card.upgraded:
                                opt_types.append(OPTION_SHOP_REMOVE)
                                opt_cards.append(vocabs.cards.get(card.id.rstrip("+")))
                                actions.append(("remove", di))

                    # Buy card options
                    for si, (sc, cost) in enumerate(zip(shop_cards, shop_costs)):
                        if sc is not None and gold >= cost:
                            opt_types.append(OPTION_SHOP_BUY)
                            opt_cards.append(vocabs.cards.get(sc.id.rstrip("+")))
                            actions.append(("buy", si, cost))

                    # Buy potion options (if we have room and gold)
                    if gold >= SHOP_POTION_COST and len(potions) < POTION_SLOTS:
                        for pi, pot in enumerate(shop_potions):
                            if pot is not None:
                                opt_types.append(OPTION_SHOP_BUY_POTION)
                                opt_cards.append(0)  # Potions aren't cards
                                actions.append(("potion", pi))

                    # Leave option (always available)
                    opt_types.append(OPTION_SHOP_LEAVE)
                    opt_cards.append(0)
                    actions.append(("leave",))

                    if len(opt_types) == 1:
                        break  # only leave available

                    with torch.no_grad():
                        hidden = network.encode_state(**st)
                        best_idx, scores = network.pick_best_option(
                            hidden, opt_types, opt_cards)

                    option_samples.append(OptionSample(
                        state_tensors={k: v.cpu() for k, v in st.items()},
                        option_types=opt_types, option_cards=opt_cards,
                        chosen_idx=best_idx, value=0.0,
                    ))

                    action = actions[best_idx]
                    if action[0] == "leave":
                        break
                    elif action[0] == "remove":
                        deck.pop(action[1])
                        gold -= SHOP_CARD_REMOVE_COST
                    elif action[0] == "buy":
                        deck.append(shop_cards[action[1]])
                        gold -= action[2]
                        shop_cards[action[1]] = None  # sold out
                    elif action[0] == "potion":
                        potions.append(dict(shop_potions[action[1]]))
                        gold -= SHOP_POTION_COST
                        shop_potions[action[1]] = None  # sold out

            except Exception:
                # Fallback to heuristic
                shop_result = _simulate_shop(deck, gold, card_db, pools, rng)
                gold += shop_result["gold_delta"]
                for idx in sorted(shop_result.get("cards_removed", []), reverse=True):
                    if idx < len(deck):
                        deck.pop(idx)
                for card in shop_result.get("cards_added", []):
                    deck.append(card)

    # Completed all floors without boss (shouldn't happen normally)
    _assign_run_values(combat_samples_by_floor, floor_reached,
                       len(room_sequence), hp, max_hp,
                       deck_change_samples, option_samples,
                       combat_hp_data=combat_hp_data,
                       boss_floors=boss_floors)
    _arch = classify_deck(deck)
    return FullRunResult(
        outcome="lose", floor_reached=floor_reached,
        final_hp=hp, max_hp=max_hp,
        combats_won=combats_won, combats_fought=combats_fought,
        deck_size=len(deck), samples=all_samples,
        deck_samples=deck_change_samples,
        option_samples=option_samples, combat_log=combat_log,
        archetype=_arch.archetype,
        archetype_commitment=_arch.commitment,
        boss_detail=boss_detail_holder,
        final_deck=[c.id for c in deck],
    )


def _assign_run_values(
    combat_samples_by_floor: dict[int, list[TrainingSample]],
    floor_reached: int,
    total_floors: int,
    final_hp: int,
    max_hp: int,
    deck_change_samples: list | None = None,
    option_samples: list | None = None,
    combat_hp_data: dict[int, tuple[int, int, int]] | None = None,
    boss_floors: set[int] | None = None,
) -> None:
    """Assign training values blending per-combat HP conservation with run outcome.

    Each combat gets a dense local signal based on how efficiently it was played
    (HP retained, potions conserved), blended with the sparse run-level outcome.
    This teaches the network that winning a combat at 5 HP is worse than at 40 HP.

    Boss fights are treated differently: HP conservation doesn't matter (HP resets
    next act), only winning and potion conservation count.
    """
    # --- Run-level value (sparse, based on overall outcome) ---
    base = floor_reached / max(1, total_floors)
    hp_bonus = final_hp / max(1, max_hp) * 0.3
    run_value = base + hp_bonus - 0.5  # [-0.5, +0.8]
    run_value = max(-1.0, min(1.0, run_value))

    # --- Per-combat values (dense, based on HP conservation) ---
    if combat_hp_data is None:
        combat_hp_data = {}
    if boss_floors is None:
        boss_floors = set()

    discount = 0.95       # run-level: earlier combats get less certain values
    turn_discount = 0.99  # within-combat temporal discount
    sorted_floors = sorted(combat_samples_by_floor.keys(), reverse=True)

    for i, floor in enumerate(sorted_floors):
        # Run-level contribution (discounted by distance from end)
        run_component = run_value * (discount ** i)

        is_boss = floor in boss_floors

        if is_boss:
            # Boss fights: winning matters most, but the loss penalty is
            # scaled by entering HP — arriving at the boss crippled (say
            # 15/80 HP) means the loss was mostly baked in by prior Act 1
            # combats, not by boss play. Teaching the network "you lost
            # with -1.0 regardless of how you got there" adds noise.
            #
            # Formula: full-HP loss = -1.0, zero-HP loss = -0.3.
            # A floor of -0.3 keeps some loss signal even when broken.
            if floor in combat_hp_data:
                hp_before, hp_after, potions_used = combat_hp_data[floor]
                if hp_after <= 0:
                    entry_ratio = max(0.0, min(1.0, hp_before / max(1, max_hp)))
                    combat_value = -(0.3 + 0.7 * entry_ratio)
                else:
                    # Won the boss fight — strong positive signal.
                    # No potion penalty: using potions at the boss is the
                    # correct strategy (they're pre-dumped for burst damage).
                    combat_value = 1.0
            else:
                combat_value = 0.0

            # Boss: weight toward win/lose outcome, less run-level blend
            blended = 0.7 * combat_value + 0.3 * run_component
        else:
            # Non-boss: HP conservation matters for surviving the run
            if floor in combat_hp_data:
                hp_before, hp_after, potions_used = combat_hp_data[floor]
                if hp_before <= 0:
                    combat_value = -1.0
                else:
                    hp_retained = hp_after / max(1, hp_before)
                    damage_fraction = (hp_before - hp_after) / max(1, max_hp)
                    potion_penalty = potions_used * 0.1
                    combat_value = hp_retained - damage_fraction * 0.5 - potion_penalty
                    combat_value = max(-1.0, min(1.0, combat_value))
            else:
                combat_value = 0.0

            blended = 0.5 * combat_value + 0.5 * run_component

        floor_samples = combat_samples_by_floor[floor]
        n = len(floor_samples)
        for j, sample in enumerate(floor_samples):
            turns_from_end = n - 1 - j
            sample.value = blended * (turn_discount ** turns_from_end)
            # Per-step penalties: wasted energy (0.15) + any block penalty
            penalty = sample.value_penalty
            if sample.wasted_energy:
                penalty += 0.15
            if penalty > 0:
                sample.value = max(-1.0, sample.value - penalty)

    # Deck change and option samples get the full run value
    if deck_change_samples:
        for sample in deck_change_samples:
            sample.value = run_value
    if option_samples:
        for sample in option_samples:
            sample.value = run_value
