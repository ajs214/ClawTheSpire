"""Collect card-pick training data by running self-play games.

Runs the existing self-play simulator (full Act 1 runs) with the current
hardcoded picker, logging every card-pick decision along with the run outcome.

Usage:
    python -m sts2_solver.collect_card_picks --games 500 --output card_pick_data.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

from .card_picker_xgb import CardPickCollector
from .data_loader import load_cards, DEFAULT_DATA_DIR
from .models import Card
from .simulator import (
    _ensure_data_loaded,
    _CHARACTERS_BY_ID,
    _normalize_card_id,
    _build_card_pool,
    _generate_act1_map,
    _offer_card_rewards,
    _pick_card_reward,
    _pick_encounter,
    _rest_site_decision,
    _simulate_event,
    _simulate_shop,
    simulate_combat,
    GOLD_REWARDS,
    POTION_DROP_CHANCE,
    POTION_SLOTS,
    POTION_TYPES,
    SHOP_CARD_REMOVE_COST,
    RunResult,
)


def collect_one_run(
    run_id: int,
    collector: CardPickCollector,
    character: str = "SILENT",
    seed: int | None = None,
    solver_time_limit_ms: float = 200.0,
    verbose: bool = False,
) -> RunResult:
    """Run one Act 1 simulation, logging card picks to the collector."""
    rng = random.Random(seed)
    random.seed(seed)

    _ensure_data_loaded()
    card_db = load_cards()

    char_data = _CHARACTERS_BY_ID.get(character, {})
    hp = char_data.get("starting_hp", 80)
    max_hp = hp
    gold = char_data.get("starting_gold", 99)
    max_energy = char_data.get("max_energy", 3)

    raw_deck_ids = char_data.get("starting_deck", [])
    deck: list[Card] = []
    for raw_id in raw_deck_ids:
        card_id = _normalize_card_id(raw_id)
        card = card_db.get(card_id)
        if card:
            deck.append(card)
        else:
            card = card_db.get(raw_id)
            if card:
                deck.append(card)

    char_color = char_data.get("color", "green")
    color_map = {"red": "ironclad", "green": "silent", "blue": "defect",
                 "purple": "necrobinder", "yellow": "regent"}
    card_color = color_map.get(char_color, char_color)
    pools = _build_card_pool(card_db, card_color)

    from .simulator import _ACTS_BY_ID
    act_data = _ACTS_BY_ID.get("OVERGROWTH", {})
    room_sequence = _generate_act1_map(rng)
    potions: list[dict] = []

    result = RunResult(run_id=run_id, outcome="lose", floor_reached=0,
                       final_hp=hp, max_hp=max_hp, gold=gold,
                       deck_size=len(deck), combats_won=0, combats_fought=0,
                       total_turns=0)

    seen_encounters: set[str] = set()
    events_list = list(act_data.get("events", []))
    rng.shuffle(events_list)
    event_idx = 0

    # Track picks for this run (to finalize later)
    run_start_idx = len(collector.records)

    for floor_num, room_type in enumerate(room_sequence, 1):
        result.floor_reached = floor_num

        if room_type in ("weak", "normal", "elite", "boss"):
            enc_id = _pick_encounter(act_data, room_type, rng, seen_encounters)
            if enc_id is None:
                continue

            combat, potions = simulate_combat(
                deck=deck, player_hp=hp, player_max_hp=max_hp,
                player_max_energy=max_energy, encounter_id=enc_id,
                card_db=card_db, rng=rng, potions=potions,
                solver_time_limit_ms=solver_time_limit_ms,
                is_boss=(room_type == "boss"),
            )
            result.combats_fought += 1
            result.total_turns += combat.turns

            if combat.outcome == "lose":
                result.outcome = "lose"
                result.death_encounter = enc_id
                result.final_hp = 0
                break

            result.combats_won += 1
            hp = combat.hp_after

            gold_range = GOLD_REWARDS.get(room_type, (10, 20))
            gold += rng.randint(*gold_range)

            if rng.random() < POTION_DROP_CHANCE and len(potions) < POTION_SLOTS:
                pot = rng.choice(POTION_TYPES)
                potions.append(dict(pot))

            if room_type != "boss":
                offered = _offer_card_rewards(pools, deck)
                pick = _pick_card_reward(offered, deck)

                # LOG THE DECISION
                collector.log_pick(
                    floor=floor_num,
                    hp=hp,
                    max_hp=max_hp,
                    deck=deck,
                    offered=offered,
                    picked=pick,
                )

                if pick:
                    deck.append(pick)
                    result.cards_picked.append(pick.name)

            if room_type == "boss":
                result.outcome = "win"

        elif room_type == "rest":
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
                hp = max(1, min(hp + changes["hp_delta"],
                                max_hp + changes["max_hp_delta"]))
                max_hp += changes["max_hp_delta"]
                gold = max(0, gold + changes["gold_delta"])
                for idx in sorted(changes["cards_removed"], reverse=True):
                    if idx < len(deck):
                        deck.pop(idx)
                for card in changes["cards_added"]:
                    deck.append(card)

        elif room_type == "shop":
            shop_result = _simulate_shop(deck, gold, card_db, pools, rng)
            gold = max(0, gold + shop_result["gold_delta"])
            for idx in sorted(shop_result["cards_removed"], reverse=True):
                if idx < len(deck):
                    deck.pop(idx)
            for card in shop_result["cards_added"]:
                deck.append(card)

    result.final_hp = hp
    result.deck_size = len(deck)

    # Finalize records for this run
    collector.finalize_run(result.outcome, result.floor_reached)

    return result


def main():
    parser = argparse.ArgumentParser(description="Collect card-pick training data")
    parser.add_argument("--games", type=int, default=500, help="Number of games to play")
    parser.add_argument("--output", type=str, default="card_pick_data.json",
                        help="Output file for card pick records")
    parser.add_argument("--character", type=str, default="SILENT")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    collector = CardPickCollector()
    wins = 0
    t0 = time.time()

    for i in range(args.games):
        seed = random.randint(0, 2**31)
        result = collect_one_run(
            run_id=i,
            collector=collector,
            character=args.character,
            seed=seed,
            verbose=args.verbose,
        )
        if result.outcome == "win":
            wins += 1

        if (i + 1) % 50 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"  Game {i+1}/{args.games} | Wins: {wins}/{i+1} "
                  f"({wins/(i+1):.1%}) | Records: {len(collector.records)} "
                  f"| {rate:.1f} games/sec")

    elapsed = time.time() - t0
    print(f"\nDone: {args.games} games in {elapsed:.1f}s")
    print(f"Win rate: {wins}/{args.games} ({wins/args.games:.1%})")
    print(f"Total card pick records: {len(collector.records)}")

    collector.save(args.output)


if __name__ == "__main__":
    main()
