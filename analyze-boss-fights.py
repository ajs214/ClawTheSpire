#!/usr/bin/env python3
"""Analyse boss_fights.jsonl produced by the training loop.

Usage:
    ./analyze-boss-fights.py                # summarise everything
    ./analyze-boss-fights.py --last 2000    # only the last N boss fights
    ./analyze-boss-fights.py --gen 500-1080 # only gens in that range

Prints:
  - overall win rate, broken down by boss id and by archetype
  - average turns, damage dealt/taken per turn
  - top cards played in boss fights (win vs lose)
  - potions used pre-combat vs during combat
  - most common loss turn (when does the bot die?)
  - decks that beat the boss vs decks that lost
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _iter_entries(path: Path, gen_range: tuple[int, int] | None, last: int | None):
    lines = path.read_text(encoding="utf-8").splitlines()
    if last is not None:
        lines = lines[-last:]
    for raw in lines:
        raw = raw.strip()
        if not raw:
            continue
        try:
            entry = json.loads(raw)
        except Exception:
            continue
        gen = entry.get("gen", 0)
        if gen_range and not (gen_range[0] <= gen <= gen_range[1]):
            continue
        yield entry


def analyse(entries):
    total = 0
    wins = 0
    by_boss = defaultdict(lambda: {"count": 0, "wins": 0})
    by_arch = defaultdict(lambda: {"count": 0, "wins": 0})
    turn_counts = []
    dmg_dealt_per_turn = []
    dmg_taken_per_turn = []
    block_per_turn = []
    cards_played_win = Counter()
    cards_played_lose = Counter()
    loss_turn = Counter()
    potions_pre = Counter()
    potions_mid = Counter()
    wasted_end_turns = 0
    total_turns = 0
    deck_size_win = []
    deck_size_lose = []
    arch_commitment_win = []
    arch_commitment_lose = []

    for entry in entries:
        boss = entry.get("boss") or {}
        boss_outcome = boss.get("outcome")
        if boss_outcome not in ("win", "lose"):
            continue
        total += 1
        is_win = boss_outcome == "win"
        wins += int(is_win)

        boss_id = "/".join(boss.get("monster_ids", []) or ["?"])
        by_boss[boss_id]["count"] += 1
        by_boss[boss_id]["wins"] += int(is_win)

        arch = entry.get("archetype", "unknown")
        by_arch[arch]["count"] += 1
        by_arch[arch]["wins"] += int(is_win)

        turn_counts.append(boss.get("total_turns", 0))
        if not is_win:
            loss_turn[boss.get("total_turns", 0)] += 1

        for pid in boss.get("precombat_potions_used", []) or []:
            potions_pre[pid] += 1

        for turn in boss.get("turns", []):
            total_turns += 1
            dmg_dealt_per_turn.append(turn.get("damage_dealt", 0))
            dmg_taken_per_turn.append(turn.get("damage_taken", 0))
            block_per_turn.append(turn.get("block_played", 0))
            if turn.get("wasted_end_turn"):
                wasted_end_turns += 1
            for cid in turn.get("cards_played", []):
                (cards_played_win if is_win else cards_played_lose)[cid] += 1
            for pid in turn.get("potions_used", []) or []:
                potions_mid[pid] += 1

        deck = entry.get("final_deck") or []
        (deck_size_win if is_win else deck_size_lose).append(len(deck))
        c = entry.get("archetype_commitment", 0.0)
        (arch_commitment_win if is_win else arch_commitment_lose).append(c)

    def pct(num, den):
        return f"{100 * num / max(1, den):.1f}%"

    def avg(xs):
        return sum(xs) / max(1, len(xs))

    print(f"=== Boss fights analysed: {total} ===")
    print(f"Overall boss win rate: {pct(wins, total)}  ({wins}/{total})")
    print()

    print("-- By boss --")
    for bid, rec in sorted(by_boss.items(), key=lambda x: -x[1]["count"]):
        print(f"  {bid:40s} n={rec['count']:5d}  wins={rec['wins']:4d}  win_rate={pct(rec['wins'], rec['count'])}")
    print()

    print("-- By archetype --")
    for arch, rec in sorted(by_arch.items(), key=lambda x: -x[1]["count"]):
        print(f"  {arch:15s} n={rec['count']:5d}  wins={rec['wins']:4d}  win_rate={pct(rec['wins'], rec['count'])}")
    print()

    print("-- Combat shape --")
    print(f"  avg turns:        {avg(turn_counts):.2f}")
    print(f"  avg dmg dealt/turn: {avg(dmg_dealt_per_turn):.2f}")
    print(f"  avg dmg taken/turn: {avg(dmg_taken_per_turn):.2f}")
    print(f"  avg block/turn:     {avg(block_per_turn):.2f}")
    if total_turns:
        print(f"  wasted end_turns:   {wasted_end_turns}/{total_turns} ({pct(wasted_end_turns, total_turns)})")
    print()

    print("-- Deck --")
    print(f"  avg deck size (win):  {avg(deck_size_win):.1f}")
    print(f"  avg deck size (lose): {avg(deck_size_lose):.1f}")
    print(f"  avg commitment (win):  {avg(arch_commitment_win):.2f}")
    print(f"  avg commitment (lose): {avg(arch_commitment_lose):.2f}")
    print()

    print("-- Potions --")
    print(f"  pre-combat (dumped at boss):")
    for pid, n in potions_pre.most_common(10):
        print(f"    {str(pid):30s} {n}")
    print(f"  mid-combat:")
    for pid, n in potions_mid.most_common(10):
        print(f"    {str(pid):30s} {n}")
    print()

    print("-- Loss turn histogram --")
    for t in sorted(loss_turn.keys()):
        bar = "#" * min(60, loss_turn[t] // max(1, sum(loss_turn.values()) // 60))
        print(f"  turn {t:2d}: {loss_turn[t]:4d} {bar}")
    print()

    print("-- Cards played in WINS (top 15) --")
    w_total = sum(cards_played_win.values())
    for cid, n in cards_played_win.most_common(15):
        print(f"  {cid:30s} {n:5d} ({pct(n, w_total)})")
    print()

    print("-- Cards played in LOSSES (top 15) --")
    l_total = sum(cards_played_lose.values())
    for cid, n in cards_played_lose.most_common(15):
        print(f"  {cid:30s} {n:5d} ({pct(n, l_total)})")
    print()

    # Cards that over-index in wins vs losses
    print("-- Cards that skew toward WINS (rate-adjusted) --")
    skew = []
    for cid in set(list(cards_played_win.keys()) + list(cards_played_lose.keys())):
        w = cards_played_win.get(cid, 0) / max(1, w_total)
        l = cards_played_lose.get(cid, 0) / max(1, l_total)
        total_count = cards_played_win.get(cid, 0) + cards_played_lose.get(cid, 0)
        if total_count >= 10:  # rare cards make poor signals
            skew.append((cid, w - l, total_count))
    skew.sort(key=lambda x: -x[1])
    for cid, d, n in skew[:10]:
        direction = "↑" if d > 0 else "↓"
        print(f"  {direction} {cid:30s} delta={d:+.4f}  (n={n})")
    print()
    print("-- Cards that skew toward LOSSES (rate-adjusted) --")
    for cid, d, n in skew[-10:][::-1]:
        direction = "↑" if d > 0 else "↓"
        print(f"  {direction} {cid:30s} delta={d:+.4f}  (n={n})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?",
                   default=str(Path(__file__).parent / "boss_fights.jsonl"))
    p.add_argument("--last", type=int, default=None,
                   help="Only analyse the last N boss fights")
    p.add_argument("--gen", type=str, default=None,
                   help="Generation range, e.g. '500-1080'")
    args = p.parse_args()

    path = Path(args.path)
    if not path.exists():
        print(f"No boss log at {path}", file=sys.stderr)
        sys.exit(1)

    gen_range = None
    if args.gen:
        lo, _, hi = args.gen.partition("-")
        gen_range = (int(lo), int(hi or lo))

    entries = list(_iter_entries(path, gen_range, args.last))
    if not entries:
        print("No matching boss-fight entries.", file=sys.stderr)
        sys.exit(1)
    analyse(entries)


if __name__ == "__main__":
    main()
