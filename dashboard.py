#!/usr/bin/env python3
"""ClawTheSpire Training Dashboard — multi-version comparison.

Monitors all training progress files, accumulates per-generation history,
persists it to dashboard_history.json, and serves a live web dashboard
comparing V1, V2, and V3 side-by-side.

Usage:
    python3 dashboard.py              # http://localhost:8090
    python3 dashboard.py --port 9000
"""

import argparse
import json
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PORT = 8090
HISTORY_FILE = SCRIPT_DIR / "dashboard_history.json"
BOSS_FLOOR = 17

# Boss-fight log files — checkpoint dirs contain boss_fights.jsonl
BOSS_LOG_DIRS = {
    "v6": "alphazero_checkpoints_v6",
    "v7": "alphazero_checkpoints_v7",
    "v8": "alphazero_checkpoints_v8",
    "v9": "alphazero_checkpoints_v9",
    "v10": "alphazero_checkpoints_v10",
    "v11": "alphazero_checkpoints_v11",
    "v12": "alphazero_checkpoints_v12",
}

# Cached boss data (refreshed by poll thread)
boss_data: dict[str, dict] = {}  # ver → {"per_boss": {id: {wins, total}}, "game_boss": {game_num: boss_id}}

# Live play run logs directory
LIVE_LOGS_DIR = SCRIPT_DIR / "logs"
live_runs: list[dict] = []  # Parsed live run summaries (refreshed by poll thread)
_live_logs_mtimes: dict[str, float] = {}  # path → mtime for change detection

def _load_boss_data_for_version(ver: str) -> dict:
    """Parse boss_fights.jsonl for a version, return per-boss stats + game→boss map."""
    dirname = BOSS_LOG_DIRS.get(ver)
    if not dirname:
        return {"per_boss": {}, "game_boss": {}}
    path = SCRIPT_DIR / dirname / "boss_fights.jsonl"
    if not path.exists():
        return {"per_boss": {}, "game_boss": {}}
    per_boss: dict[str, dict] = {}
    game_boss: dict[int, str] = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                boss_info = entry.get("boss", {})
                boss_id = boss_info.get("encounter_id", "UNKNOWN")
                game_num = entry.get("game_num", 0)
                outcome = entry.get("run_outcome", "lose")
                # Humanize: "THE_KIN_BOSS" -> "The Kin"
                game_boss[game_num] = boss_id
                if boss_id not in per_boss:
                    per_boss[boss_id] = {"wins": 0, "total": 0}
                per_boss[boss_id]["total"] += 1
                if outcome == "win":
                    per_boss[boss_id]["wins"] += 1
    except Exception:
        pass
    return {"per_boss": per_boss, "game_boss": game_boss}


def _refresh_boss_data():
    """Reload boss data for all versions."""
    global boss_data
    new_data = {}
    for ver in BOSS_LOG_DIRS:
        new_data[ver] = _load_boss_data_for_version(ver)
    with lock:
        boss_data.update(new_data)


def _parse_live_run(path: Path):
    """Parse a single JSONL run log into a summary dict for the dashboard."""
    try:
        events = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    except Exception:
        return None

    if not events:
        return None

    start = next((e for e in events if e["type"] == "run_start"), None)
    resume = next((e for e in events if e["type"] == "run_resume"), None)
    end = next((e for e in events if e["type"] == "run_end"), None)
    origin = start or resume
    if not origin:
        return None

    run_id = origin.get("run_id", "?")
    config_profile = origin.get("config_profile") or start.get("config_profile", "?") if start else "?"
    raw_checkpoint = start.get("checkpoint", "") if start else ""
    checkpoint = raw_checkpoint.rsplit("/", 1)[-1].replace(".pt", "")

    # Extract training version from checkpoint path: "alphazero_checkpoints_v11/gen_0550.pt" → "v11"
    import re as _re
    version_match = _re.search(r'_v(\d+)', raw_checkpoint)
    train_version = f"v{version_match.group(1)}" if version_match else "?"

    # Outcome and floor
    outcome = end.get("outcome", "in_progress") if end else "in_progress"
    final_floor = end.get("floor", 0) if end else 0

    # Deck evolution: starting deck + changes
    starting_deck = start.get("deck", []) if start else []
    cards_added: dict[str, int] = {}
    cards_removed: dict[str, int] = {}
    for e in events:
        if e["type"] == "deck_change":
            for card, n in (e.get("added") or {}).items():
                cards_added[card] = cards_added.get(card, 0) + n
            for card, n in (e.get("removed") or {}).items():
                cards_removed[card] = cards_removed.get(card, 0) + n
    final_deck = end.get("final_deck", []) if end else []

    # Relics
    starting_relics = start.get("relics", []) if start else []
    relics_gained = []
    for e in events:
        if e["type"] == "relic_gained":
            relics_gained.append({"name": e.get("name", "?"), "id": e.get("relic_id", "")})
    final_relics = end.get("final_relics", []) if end else []

    # Combats
    combats = []
    combat_starts = [e for e in events if e["type"] == "combat_start"]
    combat_ends = [e for e in events if e["type"] == "combat_end"]
    combat_turns = [e for e in events if e["type"] == "combat_turn"]

    for cs in combat_starts:
        floor = cs.get("floor")
        encounter_id = cs.get("encounter_id", "")
        enemies = cs.get("enemies", [])
        # Find matching end
        matching_end = next(
            (ce for ce in combat_ends if ce.get("floor") == floor),
            None,
        )
        # Find turns for this combat (between this start and next start or end)
        cs_idx = events.index(cs)
        next_cs_idx = len(events)
        for i in range(cs_idx + 1, len(events)):
            if events[i]["type"] == "combat_start":
                next_cs_idx = i
                break
        turns_in_combat = [
            e for e in events[cs_idx:next_cs_idx]
            if e["type"] == "combat_turn"
        ]
        combat_info: dict = {
            "floor": floor,
            "encounter_id": encounter_id,
            "enemies": [{"name": en.get("name", "?"), "hp": en.get("hp", 0), "max_hp": en.get("max_hp", 0)} for en in enemies],
            "turns": len(turns_in_combat),
            "cards_played": [],
        }
        for t in turns_in_combat:
            combat_info["cards_played"].extend(t.get("cards_played", []))
        if matching_end:
            combat_info["outcome"] = matching_end.get("outcome", "?")
            combat_info["hp_before"] = matching_end.get("hp_before")
            combat_info["hp_after"] = matching_end.get("hp_after")
            combat_info["is_boss"] = matching_end.get("is_boss", False)
        combats.append(combat_info)

    # Boss fights — merge boss_fight events with combat_end data.
    # boss_fight events have rich detail but sometimes empty enemies/encounter_id
    # (game clears dead enemies before the event fires). combat_end always has
    # the enemy roster from combat_start, so we cross-reference by floor.
    boss_combat_ends = {
        e.get("floor"): e for e in events
        if e["type"] == "combat_end" and e.get("is_boss")
    }
    boss_combat_starts = {
        e.get("floor"): e for e in events
        if e["type"] == "combat_start" and e.get("floor") in boss_combat_ends
    }

    boss_fights = []
    seen_floors = set()
    for e in events:
        if e["type"] == "boss_fight":
            floor = e.get("floor")
            seen_floors.add(floor)
            enemies = [
                {"name": en.get("name", "?"), "id": en.get("id", ""), "max_hp": en.get("max_hp", 0)}
                for en in e.get("enemies", [])
            ]
            # If enemies list is empty, pull from combat_end or combat_start
            if not enemies:
                ce = boss_combat_ends.get(floor) or {}
                cs = boss_combat_starts.get(floor) or {}
                source_enemies = ce.get("enemies") or cs.get("enemies") or []
                enemies = [
                    {"name": en.get("name", "?"), "id": "", "max_hp": en.get("max_hp", 0)}
                    for en in source_enemies
                ]
            # Derive boss name from enemies when encounter_id is empty
            enc_id = e.get("encounter_id", "")
            if not enc_id and enemies:
                enc_id = " + ".join(en["name"] for en in enemies)
            boss_fights.append({
                "encounter_id": enc_id,
                "floor": floor,
                "outcome": e.get("outcome", "?"),
                "turns": e.get("turns", 0),
                "hp_before": e.get("hp_before"),
                "hp_after": e.get("hp_after"),
                "enemies": enemies,
                "deck_size": e.get("deck_size", 0),
                "move_log_turns": len(e.get("move_log", [])),
            })

    # Add any boss combat_ends that didn't have a matching boss_fight event
    for floor, ce in boss_combat_ends.items():
        if floor not in seen_floors:
            enemies = ce.get("enemies", [])
            enc_id = " + ".join(en.get("name", "?") for en in enemies) if enemies else "Boss"
            boss_fights.append({
                "encounter_id": enc_id,
                "floor": floor,
                "outcome": ce.get("outcome", "?"),
                "turns": ce.get("turns", 0),
                "hp_before": ce.get("hp_before"),
                "hp_after": ce.get("hp_after"),
                "enemies": [{"name": en.get("name", "?"), "id": "", "max_hp": en.get("max_hp", 0)} for en in enemies],
                "deck_size": 0,
                "move_log_turns": 0,
            })

    # HP timeline
    hp_events = []
    for e in events:
        if e["type"] == "hp_change":
            hp_events.append({"hp": e.get("hp"), "max_hp": e.get("max_hp"), "delta": e.get("delta")})

    # Gold
    gold_events = []
    for e in events:
        if e["type"] == "gold_change":
            gold_events.append({"gold": e.get("gold"), "delta": e.get("delta")})

    # Decisions (non-auto) for card/relic selection detail
    key_decisions = []
    for e in events:
        if e["type"] == "decision" and e.get("screen_type") not in ("auto",):
            key_decisions.append({
                "screen": e.get("screen_type", "?"),
                "choice": e.get("choice", {}),
                "source": e.get("source", "?"),
                "network_value": e.get("network_value"),
            })

    ts = origin.get("ts", "")

    return {
        "run_id": run_id,
        "config_profile": config_profile,
        "checkpoint": checkpoint,
        "train_version": train_version,
        "outcome": outcome,
        "floor": final_floor,
        "starting_deck": starting_deck,
        "cards_added": cards_added,
        "cards_removed": cards_removed,
        "final_deck": final_deck,
        "final_deck_size": len(final_deck),
        "starting_relics": starting_relics,
        "relics_gained": relics_gained,
        "final_relics": final_relics,
        "combats": combats,
        "boss_fights": boss_fights,
        "hp_events": hp_events,
        "gold_events": gold_events,
        "key_decisions": key_decisions,
        "total_events": len(events),
        "ts": ts,
    }


def _refresh_live_runs():
    """Scan live play logs (recursively) and re-parse changed files."""
    global live_runs, _live_logs_mtimes
    if not LIVE_LOGS_DIR.exists():
        return
    files = sorted(LIVE_LOGS_DIR.rglob("run_*.jsonl"), key=lambda p: p.stat().st_mtime)
    changed = False
    new_mtimes = {}
    for f in files:
        mt = f.stat().st_mtime
        new_mtimes[str(f)] = mt
        if _live_logs_mtimes.get(str(f)) != mt:
            changed = True

    if not changed and len(new_mtimes) == len(_live_logs_mtimes):
        return

    # Re-parse all (could be smarter with incremental, but log count is small)
    parsed = []
    for f in files:
        run = _parse_live_run(f)
        if run:
            parsed.append(run)
    with lock:
        live_runs = parsed
        _live_logs_mtimes = new_mtimes


# Version config: label → progress file name
VERSION_FILES = {
    "v1": "alphazero_progress.json",
    "v2": "training_v2_progress.json",
    "v3": "training_v3_progress.json",
    "v4": "training_v4_progress.json",
    "v5": "training_v5_progress.json",
    "v6": "training_v6_progress.json",
    "v7": "training_v7_progress.json",
    "v8": "training_v8_progress.json",
    "v9": "training_v9_progress.json",
    "v10": "training_v10_progress.json",
    "v11": "training_v11_progress.json",
    "v12": "training_v12_progress.json",
}

# --- Shared state ---
lock = threading.Lock()
all_history: dict[str, list[dict]] = {
    "v1": [], "v2": [], "v3": [], "v4": [], "v5": [], "v6": [], "v7": [], "v8": [], "v9": [], "v10": [], "v11": [], "v12": [],
}
snapshots: dict[str, dict] = {}
active_version: str = ""  # whichever is currently training


def _load_history():
    """Load persisted history from disk."""
    global all_history
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE) as f:
                saved = json.load(f)
            for ver in all_history:
                if ver in saved and isinstance(saved[ver], list):
                    all_history[ver] = saved[ver]
            print(f"  Loaded history: " + ", ".join(
                f"{v}={len(all_history[v])} gens" for v in all_history if all_history[v]
            ))
        except Exception as e:
            print(f"  Could not load history: {e}")


def _save_history():
    """Persist history to disk."""
    try:
        tmp = HISTORY_FILE.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(all_history, f)
        tmp.replace(HISTORY_FILE)
    except Exception:
        pass


def _compute_boss_reach(recent_games: list[dict]) -> float:
    """Compute % of recent games that reached the boss floor."""
    if not recent_games:
        return 0.0
    reached = sum(1 for g in recent_games if g.get("floor", 0) >= BOSS_FLOOR)
    return reached / len(recent_games)


def _compute_boss_fight_wr(recent_games: list[dict]) -> float:
    """% of recent boss *attempts* that were won.

    A boss attempt is any game whose floor_reached >= BOSS_FLOOR.  Among
    those attempts, the ones with outcome == "win" beat the boss.  This
    tells us "when we get to the boss, how often do we actually close
    the run?" — orthogonal to boss_reach, which asks "how often do we
    even show up?".  Returns 0.0 when no recent attempts.
    """
    attempts = [g for g in recent_games if g.get("floor", 0) >= BOSS_FLOOR]
    if not attempts:
        return 0.0
    wins = sum(1 for g in attempts if g.get("outcome") == "win")
    return wins / len(attempts)


def _make_entry(data: dict) -> dict:
    """Extract a history entry from a progress snapshot."""
    recent = data.get("recent_games", [])
    # Prefer the worker-reported boss-fight WR (cumulative across the
    # whole run); fall back to computing it from the recent-games
    # window if the progress file doesn't have the field yet (older
    # training versions).
    if "boss_fight_win_rate" in data:
        boss_fight_wr = float(data.get("boss_fight_win_rate") or 0.0)
    else:
        boss_fight_wr = _compute_boss_fight_wr(recent)
    entry = {
        "generation": data.get("generation", 0),
        "games_played": data.get("games_played", 0),
        "win_rate": data.get("win_rate", 0),
        "gen_win_rate": data.get("gen_win_rate", 0),
        "policy_loss": data.get("policy_loss", 0),
        "value_loss": data.get("value_loss", 0),
        "option_loss": data.get("option_loss", 0),
        "total_loss": data.get("total_loss", 0),
        "boss_reach": _compute_boss_reach(recent),
        "boss_fight_wr": boss_fight_wr,
        "boss_fights_reached": data.get("boss_fights_reached", 0),
        "boss_fights_won": data.get("boss_fights_won", 0),
        "buffer_size": data.get("buffer_size", 0),
        "lr": data.get("lr", 0),
        "gen_time": data.get("gen_time", 0),
        "elapsed": data.get("elapsed", ""),
    }
    # Archetype stats (if present)
    arch = data.get("archetype_stats")
    if arch:
        entry["archetype_stats"] = arch
    return entry


def poll_all(interval: float = 5.0):
    """Background thread: poll all progress files, accumulate history."""
    global active_version
    last_gens: dict[str, int] = {v: -1 for v in VERSION_FILES}
    last_mtimes: dict[str, float] = {v: 0 for v in VERSION_FILES}
    save_counter = 0

    while True:
        for ver, fname in VERSION_FILES.items():
            path = SCRIPT_DIR / fname
            if not path.exists():
                continue
            try:
                mtime = path.stat().st_mtime
                if mtime == last_mtimes[ver]:
                    continue
                last_mtimes[ver] = mtime

                with open(path) as f:
                    data = json.load(f)

                with lock:
                    snapshots[ver] = data
                    # Track which version is actively training
                    if data.get("timestamp", 0) > time.time() - 30:
                        active_version = ver

                gen = data.get("generation", 0)
                if gen != last_gens[ver]:
                    last_gens[ver] = gen
                    entry = _make_entry(data)

                    with lock:
                        hist = all_history[ver]
                        # Avoid duplicates (by generation + games_played)
                        if not hist or hist[-1]["generation"] != gen or hist[-1]["games_played"] != entry["games_played"]:
                            hist.append(entry)

                    save_counter += 1
                    if save_counter % 10 == 0:
                        with lock:
                            _save_history()
            except Exception:
                pass

        # Refresh boss data and live play logs every cycle
        _refresh_boss_data()
        _refresh_live_runs()
        time.sleep(interval)


# --- HTML Dashboard ---
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ClawTheSpire Training Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0d1117; color: #c9d1d9; font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; padding: 20px; }
  h1 { color: #58a6ff; font-size: 1.4em; margin-bottom: 6px; }
  .subtitle { color: #8b949e; font-size: 0.8em; margin-bottom: 18px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .stat-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
  .stat-card .label { color: #8b949e; font-size: 0.7em; text-transform: uppercase; letter-spacing: 0.05em; }
  .stat-card .value { color: #f0f6fc; font-size: 1.5em; font-weight: bold; margin-top: 4px; }
  .stat-card .value.good { color: #3fb950; }
  .stat-card .value.warn { color: #d29922; }
  .stat-card .value.bad { color: #f85149; }
  .stat-card .sub { color: #8b949e; font-size: 0.7em; margin-top: 4px; }
  .charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 20px; }
  .chart-box { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .chart-box h2 { color: #8b949e; font-size: 0.85em; text-transform: uppercase; margin-bottom: 10px; }
  .chart-box.full { grid-column: 1 / -1; }
  .version-table { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
  .version-table h2 { color: #8b949e; font-size: 0.85em; text-transform: uppercase; margin-bottom: 10px; }
  .version-table table { width: 100%; border-collapse: collapse; }
  .version-table th { color: #8b949e; font-size: 0.7em; text-align: right; padding: 8px 10px; border-bottom: 1px solid #30363d; }
  .version-table th:first-child { text-align: left; }
  .version-table td { font-size: 0.85em; padding: 8px 10px; border-bottom: 1px solid #21262d; text-align: right; }
  .version-table td:first-child { text-align: left; font-weight: bold; }
  .v1c { color: #8b949e; }
  .v2c { color: #d29922; }
  .v3c { color: #58a6ff; }
  .v4c { color: #3fb950; }
  .earlyc { color: #6e7681; }
  .v5c { color: #bc8cff; }
  .v6c { color: #ff7b72; }
  .v7c { color: #ffa657; }
  .v8c { color: #2ea9e6; }
  .v9c { color: #39d353; }
  .v10c { color: #f0883e; }
  .v11c { color: #e05dff; }
  .v12c { color: #00d4aa; }
  .recent { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; }
  .recent h2 { color: #8b949e; font-size: 0.85em; text-transform: uppercase; margin-bottom: 10px; }
  .recent table { width: 100%; border-collapse: collapse; }
  .recent th { color: #8b949e; font-size: 0.7em; text-align: left; padding: 6px 8px; border-bottom: 1px solid #30363d; }
  .recent td { font-size: 0.85em; padding: 6px 8px; border-bottom: 1px solid #21262d; }
  .recent .win { color: #3fb950; font-weight: bold; }
  .recent .lose { color: #f85149; }
  .arch-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 20px; }
  .arch-card { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 14px; }
  .arch-card .arch-name { color: #f0f6fc; font-size: 0.95em; font-weight: bold; text-transform: capitalize; }
  .arch-card .arch-bar { background: #21262d; border-radius: 4px; height: 6px; margin: 8px 0 4px; overflow: hidden; }
  .arch-card .arch-fill { height: 100%; border-radius: 4px; transition: width 0.5s; }
  .arch-card .arch-detail { color: #8b949e; font-size: 0.7em; }
  .arch-card .arch-wr { font-size: 1.2em; font-weight: bold; margin-top: 2px; }
  .commitment-badge { display: inline-block; font-size: 0.65em; padding: 1px 5px; border-radius: 3px; margin-left: 6px; background: #21262d; color: #8b949e; }
  .bar { display: inline-block; height: 14px; border-radius: 3px; vertical-align: middle; margin-left: 6px; }
  .progress-bar { background: #21262d; border-radius: 6px; height: 8px; margin-top: 8px; overflow: hidden; }
  .progress-fill { background: linear-gradient(90deg, #1f6feb, #58a6ff); height: 100%; transition: width 0.5s; border-radius: 6px; }
  .updated { color: #484f58; font-size: 0.7em; text-align: right; margin-top: 8px; }
  @media (max-width: 900px) { .charts { grid-template-columns: 1fr; } }

  /* Live Play Section */
  .live-section { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-bottom: 20px; }
  .live-section h2 { color: #e05dff; font-size: 0.95em; text-transform: uppercase; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; }
  .live-section h2 .live-dot { width: 8px; height: 8px; border-radius: 50%; background: #3fb950; display: inline-block; animation: pulse 2s infinite; }
  @keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.3; } }
  .live-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; margin-bottom: 16px; }
  .live-stat { background: #0d1117; border: 1px solid #21262d; border-radius: 6px; padding: 10px; text-align: center; }
  .live-stat .ls-label { color: #8b949e; font-size: 0.65em; text-transform: uppercase; }
  .live-stat .ls-value { color: #f0f6fc; font-size: 1.3em; font-weight: bold; margin-top: 2px; }
  .live-runs-list table { width: 100%; border-collapse: collapse; }
  .live-runs-list th { color: #8b949e; font-size: 0.7em; text-align: left; padding: 6px 8px; border-bottom: 1px solid #30363d; }
  .live-runs-list td { font-size: 0.8em; padding: 6px 8px; border-bottom: 1px solid #21262d; cursor: pointer; }
  .live-runs-list tr:hover td { background: #1c2128; }
  .live-runs-list .run-win { color: #3fb950; font-weight: bold; }
  .live-runs-list .run-lose { color: #f85149; }
  .live-runs-list .run-progress { color: #d29922; }
  .run-detail { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 16px; margin-top: 12px; display: none; }
  .run-detail h3 { color: #58a6ff; font-size: 0.9em; margin-bottom: 10px; }
  .detail-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 900px) { .detail-grid { grid-template-columns: 1fr; } }
  .detail-panel { background: #161b22; border: 1px solid #21262d; border-radius: 6px; padding: 12px; }
  .detail-panel h4 { color: #8b949e; font-size: 0.75em; text-transform: uppercase; margin-bottom: 8px; }
  .detail-panel .card-tag { display: inline-block; background: #21262d; color: #c9d1d9; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin: 2px; }
  .detail-panel .card-tag.added { border-left: 3px solid #3fb950; }
  .detail-panel .card-tag.removed { border-left: 3px solid #f85149; text-decoration: line-through; }
  .detail-panel .relic-tag { display: inline-block; background: #21262d; color: #d29922; padding: 2px 8px; border-radius: 4px; font-size: 0.75em; margin: 2px; border-left: 3px solid #d29922; }
  .detail-panel .relic-tag.starter { color: #8b949e; border-left-color: #484f58; }
  .combat-row { padding: 6px 0; border-bottom: 1px solid #21262d; font-size: 0.8em; }
  .combat-row:last-child { border-bottom: none; }
  .combat-row .c-floor { color: #58a6ff; font-weight: bold; min-width: 40px; display: inline-block; }
  .combat-row .c-enemies { color: #c9d1d9; }
  .combat-row .c-outcome { font-weight: bold; margin-left: 8px; }
  .combat-row .c-win { color: #3fb950; }
  .combat-row .c-lose { color: #f85149; }
  .combat-row .c-cards { color: #8b949e; font-size: 0.85em; margin-top: 2px; }
  .combat-row .c-hp { color: #8b949e; font-size: 0.85em; }
  .boss-detail-box { background: #1a0d22; border: 1px solid #e05dff44; border-radius: 6px; padding: 12px; margin-top: 8px; }
  .boss-detail-box h4 { color: #e05dff; font-size: 0.8em; margin-bottom: 6px; }
</style>
</head>
<body>

<h1>&#9876; ClawTheSpire Training Dashboard</h1>
<div class="subtitle" id="active-label">Connecting...</div>

<div class="grid" id="stats"></div>

<div style="background:#161b22;border:1px solid #30363d;border-radius:8px;padding:14px;margin-bottom:20px">
  <div style="display:flex;justify-content:space-between;align-items:center">
    <span style="color:#8b949e;font-size:0.75em;text-transform:uppercase">Training Progress</span>
    <span id="gen-label" style="color:#c9d1d9;font-size:0.85em"></span>
  </div>
  <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
</div>

<div class="version-table">
  <h2>Version Comparison</h2>
  <table>
    <thead><tr><th>Version</th><th>Gens</th><th>Games</th><th>Win Rate</th><th>Boss Reach</th><th>Boss-Fight WR</th><th>Relics Used</th><th>Policy Loss</th><th>Value Loss</th><th>Total Loss</th></tr></thead>
    <tbody id="version-body"></tbody>
  </table>
</div>

<div class="version-table" id="boss-panel" style="display:none">
  <h2>Boss Win Rates (Active Version)</h2>
  <table>
    <thead><tr><th style="text-align:left">Boss</th><th>Fights</th><th>Wins</th><th>Win Rate</th><th></th></tr></thead>
    <tbody id="boss-body"></tbody>
  </table>
  <div id="boss-summary" style="color:#8b949e;font-size:0.75em;margin-top:8px"></div>
</div>

<div class="charts">
  <div class="chart-box"><h2>Policy Loss by Generation (All Versions)</h2><canvas id="policyChart"></canvas></div>
  <div class="chart-box"><h2>Value Loss by Generation (All Versions)</h2><canvas id="valueChart"></canvas></div>
  <div class="chart-box"><h2>Win Rate by Generation (All Versions)</h2><canvas id="winChart"></canvas></div>
  <div class="chart-box"><h2>Boss Reach % by Generation (All Versions)</h2><canvas id="bossChart"></canvas></div>
  <div class="chart-box"><h2>Boss-Fight Win Rate by Generation (All Versions)</h2><canvas id="bossFightChart"></canvas></div>
  <div class="chart-box"><h2>Archetype Win Rate (Active, Over Time)</h2><canvas id="archWrChart"></canvas></div>
  <div class="chart-box"><h2>Archetype Distribution (Active, Over Time)</h2><canvas id="archDistChart"></canvas></div>
  <div class="chart-box"><h2>Generation Time</h2><canvas id="timeChart"></canvas></div>
  <div class="chart-box"><h2>Floor Distribution (Active, Recent)</h2><canvas id="floorChart"></canvas></div>
</div>

<div style="margin-bottom:20px">
  <div style="color:#8b949e;font-size:0.85em;text-transform:uppercase;margin-bottom:10px">Archetype Performance (Active Version, Recent 50 Games)</div>
  <div class="arch-grid" id="arch-stats"></div>
</div>

<div class="version-table" id="relic-panel" style="display:none">
  <h2>Relic Pool <span style="color:#484f58;font-size:0.8em;font-weight:normal"> &middot; V8+ only</span></h2>
  <!-- V9 inherits the full relic pool from V8, so this panel stays visible; the "V8+ only" tag just marks the first version that emitted this telemetry. -->

  <div id="relic-summary" style="color:#8b949e;font-size:0.8em;margin-bottom:12px"></div>
  <table>
    <thead><tr><th>#</th><th>Relic</th><th>Pickups</th><th>Frequency</th><th></th></tr></thead>
    <tbody id="relic-body"></tbody>
  </table>
</div>

<div class="recent">
  <h2>Recent Games (Active Version)</h2>
  <table>
    <thead><tr><th>#</th><th>Encounter</th><th>Boss</th><th>Outcome</th><th>Floor</th><th>HP</th><th>Archetype</th><th></th></tr></thead>
    <tbody id="games-body"></tbody>
  </table>
</div>

<div class="live-section" id="live-section" style="display:none">
  <h2><span class="live-dot"></span> Live Play Performance</h2>
  <div class="live-stats" id="live-stats"></div>
  <div class="live-runs-list">
    <table>
      <thead><tr><th>Run</th><th>Version</th><th>Profile</th><th>Checkpoint</th><th>Outcome</th><th>Floor</th><th>Deck</th><th>Relics</th><th>Combats</th><th>Boss</th><th>Time</th></tr></thead>
      <tbody id="live-runs-body"></tbody>
    </table>
  </div>
  <div class="run-detail" id="run-detail"></div>
</div>

<div class="updated" id="updated"></div>

<script>
const COLORS = { early: '#6e7681', v5: '#bc8cff', v6: '#ff7b72', v7: '#ffa657', v8: '#2ea9e6', v9: '#39d353', v10: '#f0883e', v11: '#e05dff', v12: '#00d4aa' };
const LABELS = { early: 'Early (V1–V4)', v5: 'V5', v6: 'V6', v7: 'V7', v8: 'V8', v9: 'V9', v10: 'V10', v11: 'V11', v12: 'V12' };
// Display versions collapse V1–V4 into a single "early" track for readability.
const VERSIONS = ['early','v5','v6','v7','v8','v9','v10','v11','v12'];
const EARLY_SOURCES = ['v1','v2','v3','v4'];

// Concatenate V1→V4 histories in order, remapping generation numbers
// so each subsequent version's generations continue after the previous
// version's max. Produces one smooth "early training" line.
function buildEarlyHistory(histories) {
  const merged = [];
  let offset = 0;
  for (const src of EARLY_SOURCES) {
    const h = histories[src] || [];
    if (!h.length) continue;
    for (const pt of h) {
      const clone = Object.assign({}, pt);
      clone.generation = (clone.generation || 0) + offset;
      merged.push(clone);
    }
    const last = h[h.length-1];
    offset = (last.generation || 0) + offset;
  }
  return merged;
}

// Aggregate V1–V4 snapshots into one synthetic snapshot for the
// version-comparison table. Uses latest V4 values where available,
// summed totals for games/gens.
function buildEarlySnap(snapshots, histories) {
  let snap = null;
  let totalGames = 0;
  let totalGens = 0;
  for (const src of EARLY_SOURCES) {
    const s = snapshots[src];
    if (!s) continue;
    // Sum cumulative games across all training sessions for this
    // source version, not just its currently-reported session-local
    // games_played (which resets every time training is restarted).
    totalGames += computeTotalGames((histories || {})[src] || [], s.games_played || 0);
    totalGens += s.generation || 0;
    snap = s;  // keep the most recent (V4 if present)
  }
  if (!snap) return null;
  const copy = Object.assign({}, snap);
  copy.games_played = totalGames;
  copy.generation = totalGens;
  return copy;
}

// Compute cumulative games_played across all training sessions for a
// single version, by walking its persisted history and detecting
// session boundaries.
//
// games_played is session-local — it's the counter the current worker
// process has been keeping since it started, so restarting training
// resets it to ~0 even though the checkpoint and policy carry over.
// To show the user a meaningful "total games ever played by this
// version" number, we detect session boundaries in the history.
//
// A session boundary is detected when:
//   (a) games_played drops to a low value (< 200) after having been
//       much higher (> 500), AND
//   (b) the drop is sustained — the next few history entries must
//       also stay low (max of next 5 samples < 400).
//
// The sustained-drop requirement is the critical piece: it prevents
// V3's noisy history (parallel workers reporting interleaved snapshots)
// from producing ~21M ghost games via one-sample dips that aren't real
// process restarts.
function computeTotalGames(history, currentGames) {
  if (!history || !history.length) return currentGames || 0;
  const games = history.map(pt => pt.games_played || 0);
  let total = 0;
  let sessionMax = 0;
  for (let i = 0; i < games.length; i++) {
    const g = games[i];
    if (sessionMax > 500 && g < 200) {
      // Look ahead up to 5 samples — if they all stay below 400,
      // this is a real restart, not a parallel-worker jitter spike.
      const lookahead = games.slice(i, i + 5);
      if (lookahead.length >= 3) {
        let peak = 0;
        for (const v of lookahead) if (v > peak) peak = v;
        if (peak < 400) {
          total += sessionMax;
          sessionMax = 0;
        }
      }
    }
    if (g > sessionMax) sessionMax = g;
  }
  // Fold in the current live value in case the snapshot is newer than
  // the last history entry we've polled.
  const live = currentGames || 0;
  if (live > sessionMax) sessionMax = live;
  return total + sessionMax;
}

const baseOpts = {
  responsive: true,
  animation: { duration: 300 },
  plugins: { legend: { labels: { color: '#8b949e', font: { size: 11 } } } },
  scales: {
    x: { title: { display: true, text: 'Generation', color: '#484f58' }, ticks: { color: '#484f58', maxTicksLimit: 12 }, grid: { color: '#21262d' } },
    y: { ticks: { color: '#484f58' }, grid: { color: '#21262d' } },
  },
};

const pctOpts = JSON.parse(JSON.stringify(baseOpts));
pctOpts.scales.y.ticks = { ...pctOpts.scales.y.ticks, callback: function(v) { return (v*100).toFixed(0)+'%'; } };

function makeVersionDatasets(field) {
  return VERSIONS.map(v => ({
    label: LABELS[v], data: [], borderColor: COLORS[v],
    borderWidth: (v==='v12') ? 3.5 : (v==='v11') ? 3.2 : (v==='v10') ? 3 : (v==='v9') ? 2.8 : (v==='v8') ? 2.4 : (v==='v7') ? 2 : (v==='v6') ? 1.6 : (v==='v5') ? 1.4 : 1.2,
    pointRadius: 0, tension: 0.3, borderDash: v==='early' ? [4,4] : [],
  }));
}

const policyChart = new Chart(document.getElementById('policyChart'), {
  type: 'line', data: { labels: [], datasets: makeVersionDatasets('policy_loss') }, options: baseOpts,
});
const valueChart = new Chart(document.getElementById('valueChart'), {
  type: 'line', data: { labels: [], datasets: makeVersionDatasets('value_loss') }, options: baseOpts,
});
const winChart = new Chart(document.getElementById('winChart'), {
  type: 'line', data: { labels: [], datasets: makeVersionDatasets('win_rate') }, options: pctOpts,
});
const bossChart = new Chart(document.getElementById('bossChart'), {
  type: 'line', data: { labels: [], datasets: makeVersionDatasets('boss_reach') }, options: pctOpts,
});
const bossFightChart = new Chart(document.getElementById('bossFightChart'), {
  type: 'line', data: { labels: [], datasets: makeVersionDatasets('boss_fight_wr') }, options: pctOpts,
});
const timeChart = new Chart(document.getElementById('timeChart'), {
  type: 'line', data: { labels: [], datasets: [
    { label: 'Sec/Gen', data: [], borderColor: '#bc8cff', borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: { target: 'origin', above: 'rgba(188,140,255,0.08)' } },
  ]}, options: baseOpts,
});
const floorChart = new Chart(document.getElementById('floorChart'), {
  type: 'bar', data: { labels: [], datasets: [
    { label: 'Deaths', data: [], backgroundColor: '#f8514966', borderColor: '#f85149', borderWidth: 1 },
    { label: 'Wins', data: [], backgroundColor: '#3fb95066', borderColor: '#3fb950', borderWidth: 1 },
  ]}, options: { ...baseOpts, scales: { ...baseOpts.scales, x: { ...baseOpts.scales.x, title: { display: true, text: 'Floor', color: '#484f58' }, stacked: true }, y: { ...baseOpts.scales.y, stacked: true } } },
});

const ARCH_COLORS = {
  poison: '#3fb950', shiv: '#d29922', sly: '#58a6ff',
  mixed: '#bc8cff', undecided: '#484f58', unknown: '#30363d',
};

const archWrChart = new Chart(document.getElementById('archWrChart'), {
  type: 'line', data: { labels: [], datasets: [] },
  options: { ...pctOpts, plugins: { ...pctOpts.plugins, legend: { labels: { color: '#8b949e', font: { size: 11 } } } },
    scales: { ...pctOpts.scales, x: { ...pctOpts.scales.x, type: 'linear', title: { display: true, text: 'Generation', color: '#484f58' } } } },
});

const archDistChart = new Chart(document.getElementById('archDistChart'), {
  type: 'line', data: { labels: [], datasets: [] },
  options: { ...pctOpts, plugins: { ...pctOpts.plugins, legend: { labels: { color: '#8b949e', font: { size: 11 } } } },
    scales: { ...pctOpts.scales, x: { ...pctOpts.scales.x, type: 'linear', title: { display: true, text: 'Generation', color: '#484f58' } } } },
});

function thin(arr, max) {
  if (arr.length <= max) return arr;
  const step = Math.ceil(arr.length / max);
  return arr.filter((_, i) => i % step === 0 || i === arr.length - 1);
}

function updateMultiChart(chart, histories, field) {
  // Find max generation across all versions for shared x-axis
  let maxGen = 0;
  const thinned = {};
  VERSIONS.forEach((v, idx) => {
    const h = thin(histories[v] || [], 400);
    thinned[v] = h;
    if (h.length) maxGen = Math.max(maxGen, h[h.length-1].generation || 0);
  });

  // Use the version with most data points for labels
  let bestLabels = [];
  VERSIONS.forEach(v => {
    const labels = thinned[v].map(h => h.generation);
    if (labels.length > bestLabels.length) bestLabels = labels;
  });

  // For each version, map data points using their own generation numbers
  VERSIONS.forEach((v, idx) => {
    const h = thinned[v];
    chart.data.datasets[idx].data = h.map(pt => ({ x: pt.generation, y: pt[field] }));
  });

  // Use scatter-like x axis
  chart.options.scales.x.type = 'linear';
  chart.options.scales.x.min = 0;
  chart.options.scales.x.max = Math.max(maxGen, 50);
  chart.update('none');
}

function updateStats(snap, activeVer) {
  const wr = snap.win_rate || 0;
  const wrClass = wr >= 0.1 ? 'good' : wr >= 0.02 ? 'warn' : 'bad';
  const plClass = snap.policy_loss < 0.1 ? 'good' : snap.policy_loss < 0.3 ? 'warn' : 'bad';
  const recent = snap.recent_games || [];
  const reachedBoss = recent.filter(g => g.floor >= 17);
  const bossReach = recent.length ? (reachedBoss.length / recent.length * 100) : 0;
  const brClass = bossReach >= 60 ? 'good' : bossReach >= 20 ? 'warn' : 'bad';

  // Boss-fight win rate: prefer the worker-reported cumulative value,
  // then the recent-window value, then derive from recent_games.
  let bossFightWr;
  if (snap.boss_fight_win_rate != null) {
    bossFightWr = snap.boss_fight_win_rate;
  } else if (reachedBoss.length) {
    bossFightWr = reachedBoss.filter(g => g.outcome === 'win').length / reachedBoss.length;
  } else {
    bossFightWr = 0;
  }
  const recentBossFightWr = snap.recent_boss_fight_win_rate != null
    ? snap.recent_boss_fight_win_rate
    : (reachedBoss.length
        ? reachedBoss.filter(g => g.outcome === 'win').length / reachedBoss.length
        : 0);
  const bfClass = bossFightWr >= 0.25 ? 'good' : bossFightWr >= 0.05 ? 'warn' : 'bad';
  const bfReached = snap.boss_fights_reached != null ? snap.boss_fights_reached : reachedBoss.length;
  const bfWins = snap.boss_fights_won != null ? snap.boss_fights_won : reachedBoss.filter(g => g.outcome === 'win').length;

  // V8+ relic telemetry (may be missing on older versions)
  const poolSize = snap.relic_pool_size || 0;
  const uniqueSeen = snap.unique_relics_seen || 0;
  const avgRelics = snap.avg_relics_per_run;
  const relicCard = poolSize > 0
    ? `<div class="stat-card"><div class="label">Relic Pool</div><div class="value good">${poolSize}</div><div class="sub">${uniqueSeen} unique seen &middot; ${avgRelics != null ? avgRelics.toFixed(1) : '—'} /run</div></div>`
    : '';

  document.getElementById('stats').innerHTML = `
    <div class="stat-card"><div class="label">Win Rate</div><div class="value ${wrClass}">${(wr*100).toFixed(1)}%</div><div class="sub">Gen: ${((snap.gen_win_rate||0)*100).toFixed(1)}%</div></div>
    <div class="stat-card"><div class="label">Boss Reach</div><div class="value ${brClass}">${bossReach.toFixed(0)}%</div><div class="sub">Of last ${recent.length} games</div></div>
    <div class="stat-card"><div class="label">Boss-Fight WR</div><div class="value ${bfClass}">${(bossFightWr*100).toFixed(1)}%</div><div class="sub">${bfWins}/${bfReached} closed &middot; recent ${(recentBossFightWr*100).toFixed(0)}%</div></div>
    <div class="stat-card"><div class="label">Policy Loss</div><div class="value ${plClass}">${snap.policy_loss?.toFixed(4) || '—'}</div><div class="sub">Lower = smarter moves</div></div>
    <div class="stat-card"><div class="label">Value Loss</div><div class="value">${snap.value_loss?.toFixed(4) || '—'}</div><div class="sub">Position understanding</div></div>
    <div class="stat-card"><div class="label">Games Played</div><div class="value">${snap.games_played?.toLocaleString() || '—'}</div></div>
    <div class="stat-card"><div class="label">Buffer</div><div class="value">${snap.buffer_size?.toLocaleString() || '—'}</div><div class="sub">Options: ${snap.option_buffer_size?.toLocaleString() || '—'}</div></div>
    <div class="stat-card"><div class="label">Learning Rate</div><div class="value">${snap.lr?.toExponential(1) || '—'}</div></div>
    <div class="stat-card"><div class="label">Gen Time</div><div class="value">${snap.gen_time?.toFixed(1) || '—'}s</div><div class="sub">Elapsed: ${snap.elapsed || '—'}</div></div>
    ${relicCard}
  `;

  const gen = snap.generation || 0;
  const total = snap.num_generations || 1;
  const pct = (gen / total * 100).toFixed(1);
  document.getElementById('gen-label').textContent = `Gen ${gen} / ${total} (${pct}%)`;
  document.getElementById('progress-fill').style.width = pct + '%';
  document.getElementById('active-label').innerHTML =
    `Active: <span class="${activeVer}c" style="font-weight:bold">${LABELS[activeVer] || activeVer}</span> training`;
}

function updateVersionTable(snaps, histories) {
  const body = document.getElementById('version-body');
  body.innerHTML = VERSIONS.map(v => {
    const s = snaps[v];
    if (!s) return '';
    const recent = s.recent_games || [];
    const reached = recent.filter(g => g.floor >= 17);
    const br = recent.length ? (reached.length / recent.length * 100) : 0;
    // Prefer worker-reported cumulative boss-fight WR, else derive from
    // recent_games, else dash.
    let bfWr;
    if (s.boss_fight_win_rate != null) {
      bfWr = s.boss_fight_win_rate * 100;
    } else if (reached.length) {
      bfWr = reached.filter(g => g.outcome === 'win').length / reached.length * 100;
    } else {
      bfWr = null;
    }
    const bfCell = bfWr == null ? '—' : bfWr.toFixed(0) + '%';
    const totalGens = (histories[v] || []).length || s.generation || 0;
    // games_played in s is already the cumulative total across
    // training sessions — poll() pre-aggregates it with
    // computeTotalGames before calling into this function.
    // Relics-used cell: only V8+ emits relic_pool_size; older versions
    // show a dash.
    const poolSize = s.relic_pool_size || 0;
    const uniqueSeen = s.unique_relics_seen || 0;
    const relicCell = poolSize > 0
      ? `${uniqueSeen}/${poolSize} <span style="color:#8b949e">(${(uniqueSeen/poolSize*100).toFixed(0)}%)</span>`
      : '—';
    return `<tr>
      <td class="${v}c">${LABELS[v]}</td>
      <td>${totalGens}</td>
      <td>${s.games_played?.toLocaleString() || '—'}</td>
      <td>${((s.win_rate||0)*100).toFixed(1)}%</td>
      <td>${br.toFixed(0)}%</td>
      <td>${bfCell}</td>
      <td>${relicCell}</td>
      <td>${s.policy_loss?.toFixed(4) || '—'}</td>
      <td>${s.value_loss?.toFixed(4) || '—'}</td>
      <td>${s.total_loss?.toFixed(4) || '—'}</td>
    </tr>`;
  }).join('');
}

function updateFloors(games) {
  const floors = {};
  for (const g of games) {
    const f = g.floor || 0;
    if (!floors[f]) floors[f] = { wins: 0, losses: 0 };
    if (g.outcome === 'win') floors[f].wins++;
    else floors[f].losses++;
  }
  const keys = Object.keys(floors).map(Number).sort((a,b) => a-b);
  floorChart.data.labels = keys.map(f => 'F' + f);
  floorChart.data.datasets[0].data = keys.map(f => floors[f].losses);
  floorChart.data.datasets[1].data = keys.map(f => floors[f].wins);
  floorChart.update('none');
}

function humanizeBoss(id) {
  if (!id) return '—';
  // "THE_KIN_BOSS" -> "The Kin", "VANTOM_BOSS" -> "Vantom", "CEREMONIAL_BEAST_BOSS" -> "Ceremonial Beast"
  return id.replace(/_BOSS$/i, '').toLowerCase().replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function updateBossTable(bossPerBoss) {
  const panel = document.getElementById('boss-panel');
  if (!bossPerBoss || !Object.keys(bossPerBoss).length) {
    panel.style.display = 'none';
    return;
  }
  panel.style.display = '';
  const bosses = Object.entries(bossPerBoss).sort((a,b) => b[1].total - a[1].total);
  const totalFights = bosses.reduce((s, [,v]) => s + v.total, 0);
  const totalWins = bosses.reduce((s, [,v]) => s + v.wins, 0);
  const body = document.getElementById('boss-body');
  body.innerHTML = bosses.map(([id, stats]) => {
    const wr = stats.total ? (stats.wins / stats.total) : 0;
    const wrPct = (wr * 100).toFixed(1);
    const wrColor = wr >= 0.5 ? '#3fb950' : wr >= 0.3 ? '#d29922' : '#f85149';
    const barPct = Math.min(100, wr * 100);
    return `<tr>
      <td style="text-align:left;font-weight:bold">${humanizeBoss(id)}</td>
      <td>${stats.total}</td>
      <td>${stats.wins}</td>
      <td style="color:${wrColor};font-weight:bold">${wrPct}%</td>
      <td style="text-align:left"><div class="bar" style="width:${barPct}%;max-width:120px;background:${wrColor};height:10px"></div></td>
    </tr>`;
  }).join('');
  const overallWr = totalFights ? (totalWins / totalFights * 100).toFixed(1) : '0.0';
  document.getElementById('boss-summary').textContent =
    `${totalFights} total boss fights \u00b7 ${totalWins} wins \u00b7 ${overallWr}% overall`;
}

function updateGames(games, gameBossMap) {
  const body = document.getElementById('games-body');
  body.innerHTML = games.slice().reverse().map(g => {
    const cls = g.outcome === 'win' ? 'win' : 'lose';
    const pct = Math.min(100, (g.floor / 17) * 100);
    const barColor = g.outcome === 'win' ? '#3fb950' : '#f85149';
    const arch = g.archetype || '';
    const archColor = ARCH_COLORS[arch] || '#8b949e';
    const commit = g.commitment != null ? `<span class="commitment-badge">${(g.commitment*100).toFixed(0)}%</span>` : '';
    const bossId = gameBossMap[String(g.num)] || '';
    const bossName = bossId ? humanizeBoss(bossId) : (g.floor >= 17 ? '?' : '—');
    const bossColor = g.outcome === 'win' ? '#3fb950' : (g.floor >= 17 ? '#f85149' : '#484f58');
    return `<tr>
      <td>${g.num || ''}</td>
      <td>${g.encounter || ''}</td>
      <td style="color:${bossColor};font-size:0.8em">${bossName}</td>
      <td class="${cls}">${g.outcome}</td>
      <td>${g.floor}</td>
      <td>${g.hp}</td>
      <td><span style="color:${archColor};text-transform:capitalize">${arch}</span>${commit}</td>
      <td><div class="bar" style="width:${pct}px;background:${barColor}"></div></td>
    </tr>`;
  }).join('');
}

function updateArchStats(snap) {
  const archStats = snap.archetype_stats || {};
  const container = document.getElementById('arch-stats');
  const archetypes = Object.keys(archStats).sort((a,b) => (archStats[b].count||0) - (archStats[a].count||0));

  if (!archetypes.length) {
    container.innerHTML = '<div style="color:#484f58;font-size:0.8em;padding:10px">No archetype data yet — will appear after training starts with the new picker</div>';
    return;
  }

  const totalGames = archetypes.reduce((s,a) => s + (archStats[a].count||0), 0);
  container.innerHTML = archetypes.map(arch => {
    const s = archStats[arch];
    const wr = s.win_rate || 0;
    const wrColor = wr >= 0.1 ? '#3fb950' : wr >= 0.02 ? '#d29922' : '#f85149';
    const pct = totalGames ? (s.count / totalGames * 100) : 0;
    const fillColor = ARCH_COLORS[arch] || '#8b949e';
    return `<div class="arch-card">
      <div class="arch-name" style="color:${fillColor}">${arch}</div>
      <div class="arch-wr" style="color:${wrColor}">${(wr*100).toFixed(1)}% WR</div>
      <div class="arch-bar"><div class="arch-fill" style="width:${pct}%;background:${fillColor}"></div></div>
      <div class="arch-detail">${s.count} games (${pct.toFixed(0)}%) &middot; ${s.wins||0} wins</div>
    </div>`;
  }).join('');
}

function updateRelics(snap) {
  const panel = document.getElementById('relic-panel');
  const poolSize = snap.relic_pool_size || 0;
  const topRelics = snap.top_relics || [];
  if (poolSize === 0 && topRelics.length === 0) {
    panel.style.display = 'none';
    return;
  }
  panel.style.display = '';

  const uniqueSeen = snap.unique_relics_seen || 0;
  const totalPicks = snap.total_relics_picked || 0;
  const avgRelics = snap.avg_relics_per_run;
  const games = snap.games_played || 0;
  const coverage = poolSize ? (uniqueSeen / poolSize * 100).toFixed(0) : 0;

  document.getElementById('relic-summary').innerHTML =
    `<b>${poolSize}</b> simulated relics in the pool &middot; ` +
    `<b>${uniqueSeen}</b> unique seen so far (${coverage}% coverage) &middot; ` +
    `<b>${totalPicks.toLocaleString()}</b> total pickups across ${games.toLocaleString()} runs &middot; ` +
    `<b>${avgRelics != null ? avgRelics.toFixed(2) : '—'}</b> relics/run average`;

  const maxCount = topRelics.length ? topRelics[0].count : 1;
  const body = document.getElementById('relic-body');
  body.innerHTML = topRelics.slice(0, 15).map((r, i) => {
    const pct = totalPicks ? (r.count / totalPicks * 100) : 0;
    const barPct = maxCount ? (r.count / maxCount * 100) : 0;
    // Humanize relic ID: "RING_OF_THE_SNAKE" -> "Ring Of The Snake"
    const name = r.id.toLowerCase().replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
    return `<tr>
      <td>${i+1}</td>
      <td style="text-align:left">${name}</td>
      <td>${r.count}</td>
      <td>${pct.toFixed(1)}%</td>
      <td style="text-align:left"><div class="bar" style="width:${barPct}%;max-width:160px;background:#2ea9e6;height:10px"></div></td>
    </tr>`;
  }).join('');
}

function updateArchCharts(history, activeVer) {
  const h = thin(history[activeVer] || [], 300);
  // Collect all archetype names across history
  const allArchs = new Set();
  for (const pt of h) {
    if (pt.archetype_stats) Object.keys(pt.archetype_stats).forEach(a => allArchs.add(a));
  }
  const archList = [...allArchs].sort();

  // Win rate chart
  archWrChart.data.datasets = archList.map(arch => ({
    label: arch, data: h.filter(pt => pt.archetype_stats && pt.archetype_stats[arch])
      .map(pt => ({ x: pt.generation, y: pt.archetype_stats[arch].win_rate || 0 })),
    borderColor: ARCH_COLORS[arch] || '#8b949e', borderWidth: 1.5,
    pointRadius: 0, tension: 0.4,
  }));
  archWrChart.options.scales.x.max = h.length ? h[h.length-1].generation : 50;
  archWrChart.update('none');

  // Distribution chart (fraction of games per archetype)
  archDistChart.data.datasets = archList.map(arch => ({
    label: arch, data: h.filter(pt => pt.archetype_stats).map(pt => {
      const stats = pt.archetype_stats;
      const total = Object.values(stats).reduce((s,v) => s + (v.count||0), 0);
      const count = stats[arch] ? stats[arch].count : 0;
      return { x: pt.generation, y: total ? count/total : 0 };
    }),
    borderColor: ARCH_COLORS[arch] || '#8b949e', borderWidth: 1.5,
    pointRadius: 0, tension: 0.4, fill: true,
    backgroundColor: (ARCH_COLORS[arch] || '#8b949e') + '18',
  }));
  archDistChart.options.scales.x.max = h.length ? h[h.length-1].generation : 50;
  archDistChart.update('none');
}

async function poll() {
  try {
    const res = await fetch('/api/data');
    const d = await res.json();

    // --- Collapse V1–V4 into a single "early" group for display ---
    const earlyHist = buildEarlyHistory(d.history);
    const earlySnap = buildEarlySnap(d.snapshots, d.history);
    const histDisplay = { early: earlyHist };
    const snapsDisplay = {};
    if (earlySnap) snapsDisplay.early = earlySnap;
    for (const v of ['v5','v6','v7','v8','v9','v10','v11','v12']) {
      histDisplay[v] = d.history[v] || [];
      if (d.snapshots[v]) {
        // Override games_played with cumulative total across sessions
        // so every downstream consumer (stats cards, version table)
        // shows lifetime games, not just this worker process's counter.
        const copy = Object.assign({}, d.snapshots[v]);
        copy.games_played = computeTotalGames(d.history[v] || [], d.snapshots[v].games_played || 0);
        snapsDisplay[v] = copy;
      }
    }

    // Active version: if the backend reports v1–v4, map it to "early".
    let activeVer = d.active_version || 'v12';
    if (EARLY_SOURCES.includes(activeVer)) activeVer = 'early';
    const snap = snapsDisplay[activeVer] || snapsDisplay.v12 || snapsDisplay.v11 || snapsDisplay.v10 || snapsDisplay.v9 || snapsDisplay.v8 || snapsDisplay.v7 || snapsDisplay.v6 || snapsDisplay.v5 || snapsDisplay.early || {};

    updateStats(snap, activeVer);
    updateVersionTable(snapsDisplay, histDisplay);
    updateMultiChart(policyChart, histDisplay, 'policy_loss');
    updateMultiChart(valueChart, histDisplay, 'value_loss');
    updateMultiChart(winChart, histDisplay, 'win_rate');
    updateMultiChart(bossChart, histDisplay, 'boss_reach');
    updateMultiChart(bossFightChart, histDisplay, 'boss_fight_wr');

    // Time chart: active version only (use raw history for the active
    // individual version, or the collapsed early history).
    const timeSrc = activeVer === 'early' ? earlyHist : (d.history[activeVer] || []);
    const activeHist = thin(timeSrc, 300);
    timeChart.data.datasets[0].data = activeHist.map(h => ({ x: h.generation, y: h.gen_time }));
    timeChart.options.scales.x.type = 'linear';
    timeChart.update('none');

    // Boss data for active version
    const rawActiveVer = d.active_version || 'v12';
    const activeBoss = (d.boss_data || {})[rawActiveVer] || {};
    updateBossTable(activeBoss.per_boss || {});
    const gameBossMap = activeBoss.game_boss || {};

    updateFloors(snap.recent_games || []);
    updateGames(snap.recent_games || [], gameBossMap);
    updateArchStats(snap);
    // Arch charts still use the raw per-version history so early
    // archetype stats stay attached to their originating version when
    // someone clicks through; if the active version is "early", we
    // show whichever sub-version actually has archetype data.
    const archHistSrc = activeVer === 'early' ? { early: earlyHist } : d.history;
    updateArchCharts(archHistSrc, activeVer);
    updateRelics(snap);

    // Live play section
    const liveRuns = d.live_runs || [];
    window._lastLiveRuns = liveRuns;
    updateLiveRuns(liveRuns);

    document.getElementById('updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
  } catch(e) { console.error(e); }
}

// ---- Live Play Section ----
let selectedRunId = null;

function humanizeCard(name) {
  return name || '?';
}

function humanizeRelic(name) {
  if (!name) return '?';
  return name.toLowerCase().replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function formatTime(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts);
    return d.toLocaleDateString('en-US', {month:'short',day:'numeric'}) + ' ' +
           d.toLocaleTimeString('en-US', {hour:'2-digit',minute:'2-digit'});
  } catch(e) { return '—'; }
}

function updateLiveStats(runs) {
  const total = runs.length;
  const completed = runs.filter(r => r.outcome !== 'in_progress');
  const wins = runs.filter(r => r.outcome === 'win').length;
  const defeats = completed.filter(r => r.outcome !== 'win').length;
  const wr = completed.length ? (wins / completed.length * 100) : 0;
  const avgFloor = completed.length ? (completed.reduce((s,r) => s + (r.floor||0), 0) / completed.length) : 0;
  const bossReach = completed.length ? (completed.filter(r => (r.floor||0) >= 17).length / completed.length * 100) : 0;
  const profileA = runs.filter(r => r.config_profile === 'a').length;
  const profileB = runs.filter(r => r.config_profile === 'b').length;
  const totalBossFights = runs.reduce((s,r) => s + (r.boss_fights||[]).length, 0);
  const bossWins = runs.reduce((s,r) => s + (r.boss_fights||[]).filter(bf => bf.outcome === 'win').length, 0);

  const wrClass = wr >= 20 ? 'good' : wr >= 5 ? 'warn' : 'bad';
  const brClass = bossReach >= 60 ? 'good' : bossReach >= 20 ? 'warn' : 'bad';

  document.getElementById('live-stats').innerHTML = `
    <div class="live-stat"><div class="ls-label">Total Runs</div><div class="ls-value">${total}</div></div>
    <div class="live-stat"><div class="ls-label">Win Rate</div><div class="ls-value" style="color:${wr >= 20 ? '#3fb950' : wr >= 5 ? '#d29922' : '#f85149'}">${wr.toFixed(0)}%</div></div>
    <div class="live-stat"><div class="ls-label">Avg Floor</div><div class="ls-value">${avgFloor.toFixed(1)}</div></div>
    <div class="live-stat"><div class="ls-label">Boss Reach</div><div class="ls-value" style="color:${bossReach >= 50 ? '#3fb950' : '#d29922'}">${bossReach.toFixed(0)}%</div></div>
    <div class="live-stat"><div class="ls-label">Boss Fights</div><div class="ls-value">${totalBossFights} <span style="font-size:0.6em;color:#8b949e">(${bossWins}W)</span></div></div>
    <div class="live-stat"><div class="ls-label">Profile A / B</div><div class="ls-value">${profileA} / ${profileB}</div></div>
  `;
}

function updateLiveRuns(runs) {
  const section = document.getElementById('live-section');
  if (!runs || !runs.length) { section.style.display = 'none'; return; }
  section.style.display = '';

  updateLiveStats(runs);

  const body = document.getElementById('live-runs-body');
  // Show newest first
  const sorted = runs.slice().reverse();
  body.innerHTML = sorted.map(r => {
    const cls = r.outcome === 'win' ? 'run-win' : r.outcome === 'in_progress' ? 'run-progress' : 'run-lose';
    const bossInfo = (r.boss_fights || []).map(bf => bf.encounter_id || '?').join(', ') || '—';
    const bossColor = r.boss_fights && r.boss_fights.length ? (r.boss_fights.some(bf => bf.outcome === 'win') ? '#3fb950' : '#f85149') : '#484f58';
    const relicCount = (r.relics_gained || []).length + (r.starting_relics || []).length;
    const verColor = {'v9':'#39d353','v10':'#f0883e','v11':'#e05dff','v12':'#00d4aa'}[r.train_version] || '#8b949e';
    return `<tr onclick="showRunDetail('${r.run_id}')" style="${selectedRunId === r.run_id ? 'background:#1c2128' : ''}">
      <td style="font-family:monospace;color:#58a6ff">${r.run_id.substring(0,8)}</td>
      <td style="color:${verColor};font-weight:bold;font-size:0.8em">${r.train_version || '?'}</td>
      <td>${r.config_profile || '?'}</td>
      <td style="color:#8b949e;font-size:0.75em">${r.checkpoint || '—'}</td>
      <td class="${cls}">${r.outcome}</td>
      <td>${r.floor || 0}</td>
      <td>${r.final_deck_size || r.final_deck?.length || '?'}</td>
      <td>${relicCount}</td>
      <td>${(r.combats || []).length}</td>
      <td style="color:${bossColor};font-size:0.8em">${bossInfo}</td>
      <td style="color:#8b949e;font-size:0.75em">${formatTime(r.ts)}</td>
    </tr>`;
  }).join('');

  // If a run is selected, show its detail
  if (selectedRunId) {
    const run = runs.find(r => r.run_id === selectedRunId);
    if (run) renderRunDetail(run);
  }
}

function showRunDetail(runId) {
  selectedRunId = selectedRunId === runId ? null : runId;
  // Re-render from cached data
  if (window._lastLiveRuns) {
    updateLiveRuns(window._lastLiveRuns);
  }
}

function renderRunDetail(run) {
  const el = document.getElementById('run-detail');
  el.style.display = '';

  // Deck panel
  const deckHtml = buildDeckPanel(run);
  // Relics panel
  const relicHtml = buildRelicPanel(run);
  // Combats panel
  const combatHtml = buildCombatPanel(run);
  // Boss panel
  const bossHtml = buildBossPanel(run);

  el.innerHTML = `
    <h3>Run ${run.run_id} — ${run.outcome} at Floor ${run.floor}</h3>
    <div class="detail-grid">
      ${deckHtml}
      ${relicHtml}
      ${combatHtml}
      ${bossHtml}
    </div>
  `;
}

function buildDeckPanel(run) {
  const starting = (run.starting_deck || []).map(c => `<span class="card-tag">${humanizeCard(c)}</span>`).join('');
  const added = Object.entries(run.cards_added || {}).map(([c,n]) => {
    const label = n > 1 ? `${c} x${n}` : c;
    return `<span class="card-tag added">+ ${humanizeCard(label)}</span>`;
  }).join('');
  const removed = Object.entries(run.cards_removed || {}).map(([c,n]) => {
    const label = n > 1 ? `${c} x${n}` : c;
    return `<span class="card-tag removed">- ${humanizeCard(label)}</span>`;
  }).join('');
  const final = (run.final_deck || []).map(c => `<span class="card-tag">${humanizeCard(c)}</span>`).join('');

  return `<div class="detail-panel">
    <h4>Deck Evolution (${run.final_deck_size || '?'} cards)</h4>
    <div style="margin-bottom:8px"><span style="color:#8b949e;font-size:0.7em">STARTING DECK:</span><br>${starting || '<span style="color:#484f58">—</span>'}</div>
    <div style="margin-bottom:8px"><span style="color:#3fb950;font-size:0.7em">CARDS ADDED:</span><br>${added || '<span style="color:#484f58">none</span>'}</div>
    <div style="margin-bottom:8px"><span style="color:#f85149;font-size:0.7em">CARDS REMOVED:</span><br>${removed || '<span style="color:#484f58">none</span>'}</div>
    <div><span style="color:#58a6ff;font-size:0.7em">FINAL DECK:</span><br>${final || '<span style="color:#484f58">—</span>'}</div>
  </div>`;
}

function buildRelicPanel(run) {
  const starters = (run.starting_relics || []).map(r => `<span class="relic-tag starter">${r}</span>`).join('');
  const gained = (run.relics_gained || []).map(r => `<span class="relic-tag">${r.name}</span>`).join('');
  const finalR = (run.final_relics || []).map(r => `<span class="relic-tag">${r}</span>`).join('');

  return `<div class="detail-panel">
    <h4>Relics (${(run.final_relics || []).length} final)</h4>
    <div style="margin-bottom:8px"><span style="color:#8b949e;font-size:0.7em">STARTING:</span><br>${starters || '<span style="color:#484f58">—</span>'}</div>
    <div style="margin-bottom:8px"><span style="color:#d29922;font-size:0.7em">GAINED DURING RUN:</span><br>${gained || '<span style="color:#484f58">none</span>'}</div>
    <div><span style="color:#58a6ff;font-size:0.7em">FINAL RELICS:</span><br>${finalR || '<span style="color:#484f58">—</span>'}</div>
  </div>`;
}

function buildCombatPanel(run) {
  const combats = run.combats || [];
  if (!combats.length) return `<div class="detail-panel"><h4>Combats</h4><span style="color:#484f58">No combat data</span></div>`;

  const rows = combats.map(c => {
    const enemies = (c.enemies || []).map(e => `${e.name} (${e.max_hp}hp)`).join(', ');
    const outCls = c.outcome === 'win' ? 'c-win' : 'c-lose';
    const hpText = c.hp_before != null && c.hp_after != null ? `${c.hp_before} → ${c.hp_after} HP` : '';
    const cardsPlayed = (c.cards_played || []).slice(0, 12).join(', ');
    const moreCards = (c.cards_played || []).length > 12 ? ` +${c.cards_played.length - 12} more` : '';
    const bossTag = c.is_boss ? ' <span style="color:#e05dff;font-size:0.8em">[BOSS]</span>' : '';
    return `<div class="combat-row">
      <span class="c-floor">F${c.floor || '?'}</span>
      <span class="c-enemies">${enemies}</span>
      <span class="c-outcome ${outCls}">${c.outcome || '?'}</span>${bossTag}
      <span class="c-hp">${hpText}</span>
      <div style="color:#8b949e;font-size:0.75em">${c.turns || 0} turns</div>
      <div class="c-cards">${cardsPlayed}${moreCards}</div>
    </div>`;
  }).join('');

  return `<div class="detail-panel"><h4>Combats (${combats.length})</h4>${rows}</div>`;
}

function buildBossPanel(run) {
  const bossFights = run.boss_fights || [];
  if (!bossFights.length) return `<div class="detail-panel"><h4>Boss Fights</h4><span style="color:#484f58">No boss encounters</span></div>`;

  const boxes = bossFights.map(bf => {
    const enemies = (bf.enemies || []).map(e => `${e.name} (${e.max_hp}hp)`).join(', ');
    const outColor = bf.outcome === 'win' ? '#3fb950' : '#f85149';
    const hpText = bf.hp_before != null && bf.hp_after != null ? `HP: ${bf.hp_before} → ${bf.hp_after}` : '';
    return `<div class="boss-detail-box">
      <h4>${bf.encounter_id || 'Unknown Boss'}</h4>
      <div style="font-size:0.85em;color:${outColor};font-weight:bold;margin-bottom:4px">${bf.outcome?.toUpperCase()}</div>
      <div style="font-size:0.8em;color:#c9d1d9">${enemies}</div>
      <div style="font-size:0.75em;color:#8b949e;margin-top:4px">${bf.turns} turns &middot; ${hpText} &middot; Deck: ${bf.deck_size} cards &middot; ${bf.move_log_turns} turns of move data</div>
    </div>`;
  }).join('');

  return `<div class="detail-panel"><h4>Boss Fights (${bossFights.length})</h4>${boxes}</div>`;
}

poll();
setInterval(poll, 5000);
</script>
</body>
</html>"""


class DashboardHandler(SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode())
        elif self.path == "/api/data":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            with lock:
                # Build serializable boss data (convert int keys to strings for JSON)
                boss_payload = {}
                for ver, bd in boss_data.items():
                    boss_payload[ver] = {
                        "per_boss": bd.get("per_boss", {}),
                        "game_boss": {str(k): v for k, v in bd.get("game_boss", {}).items()},
                    }
                payload = {
                    "snapshots": dict(snapshots),
                    "history": {k: list(v) for k, v in all_history.items()},
                    "active_version": active_version,
                    "boss_data": boss_payload,
                    "live_runs": list(live_runs),
                }
            self.wfile.write(json.dumps(payload).encode())
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        pass


def main():
    parser = argparse.ArgumentParser(description="ClawTheSpire Training Dashboard")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    print(f"ClawTheSpire Training Dashboard")
    print(f"  Monitoring: {', '.join(VERSION_FILES.values())}")
    print(f"  History:    {HISTORY_FILE}")
    print(f"  Dashboard:  http://localhost:{args.port}")
    print()

    _load_history()

    t = threading.Thread(target=poll_all, daemon=True)
    t.start()
    time.sleep(1.5)

    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        with lock:
            _save_history()
        print("\nHistory saved. Dashboard stopped.")


if __name__ == "__main__":
    main()
