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
    "v13": "alphazero_checkpoints_v13",
    "v14": "alphazero_checkpoints_v14",
    "v15": "alphazero_checkpoints_v15",
    "v16": "alphazero_checkpoints_v16",
    "v17": "alphazero_checkpoints_v17",
    "v18": "alphazero_checkpoints_v18",
}

# Cached boss data (refreshed by poll thread)
boss_data: dict[str, dict] = {}  # ver → {"per_boss": {id: {wins, total}}, "game_boss": {game_num: boss_id}}

# Live play run logs directory
LIVE_LOGS_DIR = SCRIPT_DIR / "logs"
live_runs: list[dict] = []  # Parsed live run summaries (refreshed by poll thread)
_live_logs_mtimes: dict[str, float] = {}  # path → mtime for change detection

def _load_boss_data_for_version(ver: str) -> dict:
    """Parse boss_fights.jsonl for a version, return per-boss stats + game→boss map + gen timeline."""
    dirname = BOSS_LOG_DIRS.get(ver)
    if not dirname:
        return {"per_boss": {}, "game_boss": {}, "gen_wr": []}
    path = SCRIPT_DIR / dirname / "boss_fights.jsonl"
    if not path.exists():
        return {"per_boss": {}, "game_boss": {}, "gen_wr": []}
    per_boss: dict[str, dict] = {}
    game_boss: dict[int, str] = {}
    # Collect all fights with gen for timeline chart
    all_fights: list[dict] = []
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
                gen = entry.get("gen", 0)
                outcome = entry.get("run_outcome", "lose")
                game_boss[game_num] = boss_id
                if boss_id not in per_boss:
                    per_boss[boss_id] = {"wins": 0, "total": 0}
                per_boss[boss_id]["total"] += 1
                if outcome == "win":
                    per_boss[boss_id]["wins"] += 1
                all_fights.append({"gen": gen, "boss_id": boss_id, "won": outcome == "win"})
    except Exception:
        pass

    # Build per-generation win rate timeline (rolling window of 50 fights)
    gen_wr = _compute_gen_win_rate_timeline(all_fights, window=50)

    return {"per_boss": per_boss, "game_boss": game_boss, "gen_wr": gen_wr}


def _compute_gen_win_rate_timeline(fights: list[dict], window: int = 50) -> list[dict]:
    """Compute rolling win rate over generations, overall and per-boss.

    Returns a list of {gen, overall, <boss_id>: wr, ...} dicts, one per
    window step. Uses a sliding window of `window` fights, stepping by
    window//2 for overlap.
    """
    if not fights:
        return []
    fights.sort(key=lambda f: f["gen"])
    step = max(1, window // 2)
    result = []
    boss_ids = sorted(set(f["boss_id"] for f in fights))

    for start in range(0, len(fights), step):
        chunk = fights[start:start + window]
        if len(chunk) < 5:  # Skip tiny windows
            continue
        mid_gen = chunk[len(chunk) // 2]["gen"]
        wins = sum(1 for f in chunk if f["won"])
        entry: dict = {"gen": mid_gen, "overall": wins / len(chunk)}
        for bid in boss_ids:
            boss_chunk = [f for f in chunk if f["boss_id"] == bid]
            if boss_chunk:
                entry[bid] = sum(1 for f in boss_chunk if f["won"]) / len(boss_chunk)
        result.append(entry)
    return result


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
    "v13": "training_v13_progress.json",
    "v14": "training_v14_progress.json",
    "v15": "training_v15_progress.json",
    "v16": "training_v16_progress.json",
    "v17": "training_v17_progress.json",
    "v18": "training_v18_progress.json",
}

# --- Shared state ---
lock = threading.Lock()
all_history: dict[str, list[dict]] = {
    "v1": [], "v2": [], "v3": [], "v4": [], "v5": [], "v6": [], "v7": [], "v8": [], "v9": [], "v10": [], "v11": [], "v12": [], "v13": [], "v14": [], "v15": [], "v16": [], "v17": [], "v18": [],
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
        "card_pick_loss": data.get("card_pick_loss", 0),
        "other_option_loss": data.get("other_option_loss", 0),
        "card_pick_agreement": data.get("card_pick_agreement", 0),
        "card_skip_rate": data.get("card_skip_rate", 0),
        "card_pick_score_spread": data.get("card_pick_score_spread", 0),
        "total_loss": data.get("total_loss", 0),
        "boss_reach": _compute_boss_reach(recent),
        "boss_fight_wr": boss_fight_wr,
        "boss_fights_reached": data.get("boss_fights_reached", 0),
        "boss_fights_won": data.get("boss_fights_won", 0),
        "games_per_gen": data.get("games_per_gen", 8),
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
        now = time.time()
        # Collect all versions with recent timestamps, then pick the
        # highest version number so V13 wins over V12 when both are running.
        recently_active: list[str] = []
        for ver, fname in VERSION_FILES.items():
            path = SCRIPT_DIR / fname
            if not path.exists():
                continue
            try:
                mtime = path.stat().st_mtime
                if mtime == last_mtimes[ver]:
                    # File unchanged — still check if its snapshot is recent
                    if snapshots.get(ver, {}).get("timestamp", 0) > now - 60:
                        recently_active.append(ver)
                    continue
                last_mtimes[ver] = mtime

                with open(path) as f:
                    data = json.load(f)

                with lock:
                    snapshots[ver] = data
                    if data.get("timestamp", 0) > now - 60:
                        recently_active.append(ver)

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

        # Pick the highest version number among recently active versions.
        # This ensures V13 wins over V12 when both are running. If nothing
        # is active, clear it so the JS falls back to the newest with data.
        with lock:
            if recently_active:
                # Sort by version number descending, pick highest
                def _ver_num(v: str) -> int:
                    try:
                        return int(v.lstrip("v"))
                    except ValueError:
                        return 0
                active_version = max(recently_active, key=_ver_num)
            else:
                active_version = ""

        # Refresh boss data and live play logs every cycle
        _refresh_boss_data()
        _refresh_live_runs()
        time.sleep(interval)


# --- HTML Dashboard ---
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Andrew's ClawTheSpire Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#0d1117; color:#c9d1d9; font-family:'SF Mono','Fira Code','Consolas',monospace; padding:20px; }
  h1 { color:#58a6ff; font-size:1.4em; margin-bottom:4px; }
  .subtitle { color:#8b949e; font-size:0.8em; margin-bottom:14px; }

  /* Tabs */
  .tab-bar { display:flex; gap:0; margin-bottom:20px; border-bottom:2px solid #30363d; }
  .tab-btn { background:none; border:none; color:#8b949e; font-family:inherit; font-size:0.85em; padding:10px 20px; cursor:pointer; border-bottom:2px solid transparent; margin-bottom:-2px; transition:all 0.2s; }
  .tab-btn:hover { color:#c9d1d9; }
  .tab-btn.active { color:#58a6ff; border-bottom-color:#58a6ff; font-weight:bold; }
  .tab-panel { display:none; }
  .tab-panel.active { display:block; }

  /* Cards */
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin-bottom:20px; }
  .card { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:14px; }
  .card .label { color:#8b949e; font-size:0.7em; text-transform:uppercase; letter-spacing:0.05em; }
  .card .value { color:#f0f6fc; font-size:1.5em; font-weight:bold; margin-top:4px; }
  .card .value.good { color:#3fb950; } .card .value.warn { color:#d29922; } .card .value.bad { color:#f85149; }
  .card .sub { color:#8b949e; font-size:0.7em; margin-top:4px; }

  /* Section panels */
  .panel { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:20px; }
  .panel h2 { color:#8b949e; font-size:0.85em; text-transform:uppercase; margin-bottom:10px; }
  .panel table { width:100%; border-collapse:collapse; }
  .panel th { color:#8b949e; font-size:0.7em; text-align:right; padding:8px 10px; border-bottom:1px solid #30363d; }
  .panel th:first-child { text-align:left; }
  .panel td { font-size:0.85em; padding:8px 10px; border-bottom:1px solid #21262d; text-align:right; }
  .panel td:first-child { text-align:left; font-weight:bold; }

  /* Charts */
  .charts { display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-bottom:20px; }
  .chart-box { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; }
  .chart-box h2 { color:#8b949e; font-size:0.85em; text-transform:uppercase; margin-bottom:10px; }
  .chart-box.full { grid-column:1/-1; }
  @media (max-width:900px) { .charts { grid-template-columns:1fr; } }

  /* Version colors */
  .earlyc{color:#6e7681} .v5c{color:#bc8cff} .v6c{color:#ff7b72} .v7c{color:#ffa657}
  .v8c{color:#2ea9e6} .v9c{color:#39d353} .v10c{color:#f0883e} .v11c{color:#e05dff} .v12c{color:#00d4aa} .v13c{color:#ff6b9d} .v14c{color:#79c0ff} .v15c{color:#ffd700} .v16c{color:#ff6347} .v17c{color:#7b68ee}

  /* Boss table */
  .boss-row td { text-align:left; }
  .bar { display:inline-block; height:10px; border-radius:3px; vertical-align:middle; }

  /* Progress bar */
  .progress-bar { background:#21262d; border-radius:6px; height:8px; margin-top:8px; overflow:hidden; }
  .progress-fill { background:linear-gradient(90deg,#1f6feb,#58a6ff); height:100%; transition:width 0.5s; border-radius:6px; }

  /* Recent games */
  .win { color:#3fb950; font-weight:bold; } .lose { color:#f85149; }
  .act1win { color:#3fb950; }

  /* Live play */
  .live-stat { background:#0d1117; border:1px solid #21262d; border-radius:6px; padding:10px; text-align:center; }
  .live-stat .ls-label { color:#8b949e; font-size:0.65em; text-transform:uppercase; }
  .live-stat .ls-value { color:#f0f6fc; font-size:1.3em; font-weight:bold; margin-top:2px; }
  .live-runs-list table { width:100%; border-collapse:collapse; }
  .live-runs-list th { color:#8b949e; font-size:0.7em; text-align:left; padding:6px 8px; border-bottom:1px solid #30363d; }
  .live-runs-list td { font-size:0.8em; padding:6px 8px; border-bottom:1px solid #21262d; cursor:pointer; }
  .live-runs-list tr:hover td { background:#1c2128; }

  /* Detail panels */
  .run-detail { background:#0d1117; border:1px solid #30363d; border-radius:8px; padding:16px; margin-top:12px; display:none; }
  .run-detail h3 { color:#58a6ff; font-size:0.9em; margin-bottom:10px; }
  .detail-grid { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
  @media (max-width:900px) { .detail-grid { grid-template-columns:1fr; } }
  .detail-panel { background:#161b22; border:1px solid #21262d; border-radius:6px; padding:12px; }
  .detail-panel h4 { color:#8b949e; font-size:0.75em; text-transform:uppercase; margin-bottom:8px; }
  .detail-panel .card-tag { display:inline-block; background:#21262d; color:#c9d1d9; padding:2px 8px; border-radius:4px; font-size:0.75em; margin:2px; }
  .detail-panel .card-tag.added { border-left:3px solid #3fb950; }
  .detail-panel .card-tag.removed { border-left:3px solid #f85149; text-decoration:line-through; }
  .detail-panel .relic-tag { display:inline-block; background:#21262d; color:#d29922; padding:2px 8px; border-radius:4px; font-size:0.75em; margin:2px; border-left:3px solid #d29922; }
  .detail-panel .relic-tag.starter { color:#8b949e; border-left-color:#484f58; }
  .combat-row { padding:6px 0; border-bottom:1px solid #21262d; font-size:0.8em; }
  .combat-row:last-child { border-bottom:none; }
  .combat-row .c-floor { color:#58a6ff; font-weight:bold; min-width:40px; display:inline-block; }
  .combat-row .c-outcome { font-weight:bold; margin-left:8px; }
  .combat-row .c-win { color:#3fb950; } .combat-row .c-lose { color:#f85149; }
  .combat-row .c-cards { color:#8b949e; font-size:0.85em; margin-top:2px; }
  .boss-detail-box { background:#1a0d22; border:1px solid #e05dff44; border-radius:6px; padding:12px; margin-top:8px; }
  .boss-detail-box h4 { color:#e05dff; font-size:0.8em; margin-bottom:6px; }

  .updated { color:#484f58; font-size:0.7em; text-align:right; margin-top:8px; }
  .section-label { color:#8b949e; font-size:0.75em; text-transform:uppercase; margin-bottom:8px; letter-spacing:0.05em; }
  .commitment-badge { display:inline-block; font-size:0.65em; padding:1px 5px; border-radius:3px; margin-left:6px; background:#21262d; color:#8b949e; }
</style>
</head>
<body>

<h1>&#9876; Andrew's ClawTheSpire Dashboard</h1>
<div class="subtitle" id="active-label">Connecting...</div>

<div class="tab-bar">
  <button class="tab-btn active" onclick="switchTab('summary')">Summary</button>
  <button class="tab-btn" onclick="switchTab('training')">Training</button>
  <button class="tab-btn" onclick="switchTab('live')">Live Play</button>
</div>

<!-- ==================== SUMMARY TAB ==================== -->
<div class="tab-panel active" id="tab-summary">

  <div class="section-label" id="summary-version-label">Training — Whole Version</div>
  <div class="grid" id="summary-version-stats"></div>

  <div class="section-label">Training — Current Run</div>
  <div class="grid" id="summary-run-stats"></div>

  <div class="section-label">Training — Last 10 Generations</div>
  <div class="grid" id="summary-recent-stats"></div>

  <div class="section-label">Card Selector (card_eval_head)</div>
  <div class="grid" id="summary-card-pick-stats"></div>

  <div class="section-label">Training — Boss Win Rates</div>
  <div class="panel" id="summary-boss-panel" style="margin-bottom:20px">
    <table>
      <thead><tr><th style="text-align:left">Boss</th><th>Cumulative</th><th>Last 10 Gens</th><th></th></tr></thead>
      <tbody id="summary-boss-body"></tbody>
    </table>
  </div>

  <div class="section-label">Live Play — Current Version</div>
  <div class="grid" id="summary-live-stats"></div>

  <div class="section-label">Training Progress</div>
  <div class="card" style="margin-bottom:20px">
    <div style="display:flex;justify-content:space-between;align-items:center">
      <span id="progress-label" style="color:#c9d1d9;font-size:0.85em"></span>
      <span id="progress-elapsed" style="color:#8b949e;font-size:0.8em"></span>
    </div>
    <div class="progress-bar"><div class="progress-fill" id="progress-fill"></div></div>
  </div>
</div>

<!-- ==================== TRAINING TAB ==================== -->
<div class="tab-panel" id="tab-training">

  <div class="panel">
    <h2>Version Comparison</h2>
    <table>
      <thead><tr><th>Version</th><th>Gens</th><th>Games</th><th>Win Rate</th><th>Boss Reach</th><th>Boss-Fight WR</th><th>Card Agree</th><th>Card Spread</th><th>Policy Loss</th><th>Value Loss</th><th>Card Pick Loss</th><th>Total Loss</th></tr></thead>
      <tbody id="version-body"></tbody>
    </table>
  </div>

  <div class="charts">
    <div class="chart-box full"><h2>Win Rate by Generation (All Lineages)</h2><canvas id="winChart"></canvas></div>
    <div class="chart-box full"><h2>Win Rate by Generation — Per Boss (Active Version)</h2><canvas id="genBossWrChart"></canvas></div>
    <div class="chart-box full"><h2>Card Selector — Agreement & Score Spread (Active Version)</h2><canvas id="cardPickChart"></canvas></div>
  </div>

  <div class="panel">
    <h2>Card Preferences (Active Version — Cumulative)</h2>
    <table>
      <thead><tr><th style="text-align:left">Card</th><th>Offered</th><th>Picked</th><th>Pick Rate</th><th>Skip Rate</th><th>Win Pick Rate</th><th></th></tr></thead>
      <tbody id="card-stats-body"></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Recent Games (Active Version)</h2>
    <table>
      <thead><tr><th>#</th><th>Encounter</th><th>Boss</th><th>Outcome</th><th>Floor</th><th>HP</th><th>Archetype</th><th>Picks</th><th>Agree</th><th>Spread</th><th></th></tr></thead>
      <tbody id="games-body"></tbody>
    </table>
  </div>
</div>

<!-- ==================== LIVE PLAY TAB ==================== -->
<div class="tab-panel" id="tab-live">

  <div class="grid" id="live-stats"></div>

  <div class="panel" id="live-boss-panel">
    <h2>Boss Win Rates (Live Play — Current Version)</h2>
    <table>
      <thead><tr><th style="text-align:left">Boss</th><th>Fights</th><th>Wins</th><th>Win Rate</th><th></th></tr></thead>
      <tbody id="live-boss-body"></tbody>
    </table>
  </div>

  <div class="panel">
    <h2>Live Runs</h2>
    <div class="live-runs-list">
      <table>
        <thead><tr><th>Version</th><th>Profile</th><th>Checkpoint</th><th>Outcome</th><th>Floor</th><th>Deck</th><th>Relics</th><th>Combats</th><th>Boss</th><th>Time</th></tr></thead>
        <tbody id="live-runs-body"></tbody>
      </table>
    </div>
    <div class="run-detail" id="run-detail"></div>
  </div>
</div>

<div class="updated" id="updated"></div>

<script>
const COLORS = { early:'#6e7681', v1:'#6e7681', v2:'#7c858d', v3:'#8a939a', v4:'#99a1a8', v5:'#bc8cff', v6:'#ff7b72', v7:'#ffa657', v8:'#2ea9e6', v9:'#39d353', v10:'#f0883e', v11:'#e05dff', v12:'#00d4aa', v13:'#ff6b9d', v14:'#79c0ff', v15:'#ffd700', v16:'#ff6347', v17:'#7b68ee', v18:'#00ff88' };
const LABELS = { early:'Early (V1-V4)', v1:'V1', v2:'V2', v3:'V3', v4:'V4', v5:'V5', v6:'V6', v7:'V7', v8:'V8', v9:'V9', v10:'V10', v11:'V11', v12:'V12', v13:'V13', v14:'V14', v15:'V15', v16:'V16', v17:'V17', v18:'V18' };
const VERSIONS = ['early','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17','v18'];
const EARLY_SOURCES = ['v1','v2','v3','v4'];
// Last 5 display versions for the win rate chart (legacy, kept for version table)
const RECENT_VERSIONS = ['v9','v10','v11','v12','v13','v14','v15','v16','v17','v18'];
const BOSS_FLOOR = 17;

// ---- Lineage definitions ----
// Each lineage is a cold-start chain. Versions within a lineage are warm
// continuations — their generation counts are stitched end-to-end.
const LINEAGES = [
  { id: 'L1', label: 'V2\u2013V11', versions: ['v2','v3','v4','v5','v6','v7','v8','v9','v10','v11'] },
  { id: 'L2', label: 'V12',       versions: ['v12'] },
  { id: 'L3', label: 'V13',       versions: ['v13'] },
  { id: 'L4', label: 'V14',       versions: ['v14'] },
  { id: 'L5', label: 'V15',       versions: ['v15'] },
  { id: 'L6', label: 'V16–V18', versions: ['v16','v17','v18'] },
];

const ARCH_COLORS = { poison:'#3fb950', shiv:'#d29922', sly:'#58a6ff', mixed:'#bc8cff', undecided:'#484f58', unknown:'#30363d' };
const BOSS_CHART_COLORS = { 'OVERALL':'#f0f6fc', 'VANTOM_BOSS':'#58a6ff', 'THE_KIN_BOSS':'#d29922', 'CEREMONIAL_BEAST_BOSS':'#ff7b72' };

// ---- Tab switching ----
function switchTab(name) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('tab-' + name).classList.add('active');
  document.querySelector(`.tab-btn[onclick="switchTab('${name}')"]`).classList.add('active');
  // Trigger chart resize after tab switch (hidden canvases have 0 size)
  setTimeout(() => { window.dispatchEvent(new Event('resize')); }, 50);
}

// ---- Utility ----
function buildEarlyHistory(histories) {
  const merged = []; let offset = 0;
  for (const src of EARLY_SOURCES) {
    const h = histories[src] || []; if (!h.length) continue;
    for (const pt of h) { const c = Object.assign({}, pt); c.generation = (c.generation||0)+offset; merged.push(c); }
    offset = (h[h.length-1].generation||0) + offset;
  }
  return merged;
}

function buildEarlySnap(snapshots, histories) {
  let snap = null, totalGames = 0, totalGens = 0;
  for (const src of EARLY_SOURCES) {
    const s = snapshots[src]; if (!s) continue;
    totalGames += computeTotalGames((histories||{})[src]||[], s.games_played||0);
    totalGens += s.generation||0; snap = s;
  }
  if (!snap) return null;
  const copy = Object.assign({}, snap); copy.games_played = totalGames; copy.generation = totalGens; return copy;
}

function computeTotalGames(history, currentGames) {
  if (!history || !history.length) return currentGames || 0;
  const games = history.map(pt => pt.games_played || 0);
  let total = 0, sessionMax = 0;
  for (let i = 0; i < games.length; i++) {
    const g = games[i];
    if (sessionMax > 500 && g < 200) {
      const la = games.slice(i, i+5);
      if (la.length >= 3) { let pk=0; for(const v of la) if(v>pk)pk=v; if(pk<400){total+=sessionMax;sessionMax=0;} }
    }
    if (g > sessionMax) sessionMax = g;
  }
  const live = currentGames||0; if(live>sessionMax) sessionMax=live;
  return total + sessionMax;
}

function thin(arr, max) {
  if (arr.length <= max) return arr;
  const step = Math.ceil(arr.length / max);
  return arr.filter((_, i) => i % step === 0 || i === arr.length - 1);
}

// Stitch together history segments from multiple training restarts.
// Each restart resets the generation counter to 1, creating disconnected
// segments.  This function adds a cumulative offset so the chart shows
// one continuous line per version.
function stitchGenerations(history) {
  if (!history || !history.length) return [];
  const out = [];
  let offset = 0;
  let prevGen = 0;
  for (const pt of history) {
    const g = pt.generation || 0;
    // Detect reset: generation dropped significantly
    if (g < prevGen - 5) {
      offset += prevGen;
    }
    prevGen = g;
    out.push(Object.assign({}, pt, { generation: g + offset }));
  }
  return out;
}

function humanizeBoss(id) {
  if (!id) return '-';
  return id.replace(/_BOSS$/i,'').toLowerCase().replace(/_/g,' ').replace(/\b\w/g, c=>c.toUpperCase());
}

function formatTime(ts) {
  if (!ts) return '-';
  try { const d=new Date(ts); return d.toLocaleDateString('en-US',{month:'short',day:'numeric'})+' '+d.toLocaleTimeString('en-US',{hour:'2-digit',minute:'2-digit'}); }
  catch(e) { return '-'; }
}

function wrColor(wr) { return wr >= 0.25 ? '#3fb950' : wr >= 0.05 ? '#d29922' : '#f85149'; }
function wrClass(wr) { return wr >= 0.1 ? 'good' : wr >= 0.02 ? 'warn' : 'bad'; }

// ---- Chart setup ----
const baseOpts = {
  responsive:true, animation:{duration:300},
  plugins:{legend:{labels:{color:'#8b949e',font:{size:11}}}},
  scales:{
    x:{type:'linear',title:{display:true,text:'Generation',color:'#484f58'},ticks:{color:'#484f58',maxTicksLimit:12},grid:{color:'#21262d'}},
    y:{ticks:{color:'#484f58'},grid:{color:'#21262d'}},
  },
};
const pctOpts = JSON.parse(JSON.stringify(baseOpts));
pctOpts.scales.y.ticks = {...pctOpts.scales.y.ticks, callback:function(v){return (v*100).toFixed(0)+'%';}};

// Win rate chart — lineage-based (one continuous line per cold start)
const winChart = new Chart(document.getElementById('winChart'), {
  type:'line', data:{labels:[],datasets:[]}, options:pctOpts,
});

// Per-boss WR chart
const genBossWrChart = new Chart(document.getElementById('genBossWrChart'), {
  type:'line', data:{labels:[],datasets:[]},
  options:{...pctOpts, plugins:{...pctOpts.plugins,legend:{labels:{color:'#8b949e',font:{size:11}}}},
    scales:{...pctOpts.scales,x:{...pctOpts.scales.x,type:'linear',title:{display:true,text:'Generation',color:'#484f58'}}}},
});

// Card-pick dual-axis chart: agreement (left, %) + score spread (right, raw)
const cardPickChart = new Chart(document.getElementById('cardPickChart'), {
  type:'line', data:{labels:[],datasets:[]},
  options:{
    responsive:true, animation:{duration:300},
    plugins:{legend:{labels:{color:'#8b949e',font:{size:11}}}},
    scales:{
      x:{type:'linear',title:{display:true,text:'Generation',color:'#484f58'},ticks:{color:'#484f58',maxTicksLimit:12},grid:{color:'#21262d'}},
      yAgree:{position:'left',title:{display:true,text:'Agreement %',color:'#58a6ff'},ticks:{color:'#58a6ff',callback:function(v){return (v*100).toFixed(0)+'%';}},grid:{color:'#21262d'}},
      ySpread:{position:'right',title:{display:true,text:'Score Spread',color:'#d29922'},ticks:{color:'#d29922'},grid:{drawOnChartArea:false}},
    },
  },
});

// ---- Update functions ----

function computeVersionStats(history, snap) {
  // Compute whole-version stats by summing per-gen wins across all runs.
  // Detect run boundaries where games_played drops (new run started).
  let totalGames = 0, totalWins = 0, totalBossReached = 0, totalBossWon = 0;
  let prevGamesPlayed = 0;
  for (const h of history) {
    const gp = h.games_played || 0;
    const gpg = h.games_per_gen || 8;
    // Detect run boundary: games_played dropped significantly
    if (gp < prevGamesPlayed * 0.5 && prevGamesPlayed > 50) {
      // Previous run ended at prevGamesPlayed; stats already accumulated
    }
    prevGamesPlayed = gp;
    // Per-gen contribution
    const genGames = gpg;
    const genWins = (h.gen_win_rate || 0) * genGames;
    totalGames += genGames;
    totalWins += genWins;
  }
  // Boss fight stats: use boss_fights_reached/won from the latest history entry
  // across run boundaries. Since these are per-run cumulative, we need to detect
  // run boundaries and sum the peaks.
  let bfReached = 0, bfWon = 0, prevBfR = 0, prevBfW = 0;
  for (const h of history) {
    const r = h.boss_fights_reached || 0;
    const w = h.boss_fights_won || 0;
    if (r < prevBfR * 0.5 && prevBfR > 5) {
      // Run boundary — bank previous run's totals
      bfReached += prevBfR;
      bfWon += prevBfW;
    }
    prevBfR = r;
    prevBfW = w;
  }
  // Add current run's values (from snapshot, which is most up-to-date)
  bfReached += (snap.boss_fights_reached || prevBfR || 0);
  bfWon += (snap.boss_fights_won || prevBfW || 0);

  const wr = totalGames ? totalWins / totalGames : 0;
  const bfWr = bfReached ? bfWon / bfReached : 0;
  return { totalGames, totalWins, wr, bfReached, bfWon, bfWr };
}

function updateSummaryTab(snap, activeVer, bossPerBoss, liveRuns, recentHistory) {
  // === Section 1: Whole Version (all runs combined) ===
  const vs = computeVersionStats(recentHistory, snap);
  document.getElementById('summary-version-label').textContent =
    `Training \u2014 Whole Version (${LABELS[activeVer]||activeVer})`;
  document.getElementById('summary-version-stats').innerHTML = `
    <div class="card"><div class="label">Act 1 Win Rate</div><div class="value ${wrClass(vs.wr)}">${(vs.wr*100).toFixed(1)}%</div>
      <div class="sub">${vs.totalWins.toFixed(0)} wins / ${vs.totalGames} games</div></div>
    <div class="card"><div class="label">Boss-Fight Win Rate</div><div class="value ${wrClass(vs.bfWr)}">${(vs.bfWr*100).toFixed(1)}%</div>
      <div class="sub">${vs.bfWon}/${vs.bfReached} boss fights won</div></div>
    <div class="card"><div class="label">Total Generations</div><div class="value" style="color:#58a6ff">${recentHistory.length}</div>
      <div class="sub">across all runs</div></div>
  `;

  // === Section 2: Current Run ===
  const runWr = snap.win_rate||0;
  const runBfWr = snap.boss_fight_win_rate||0;
  const runBfReached = snap.boss_fights_reached||0;
  const runBfWon = snap.boss_fights_won||0;
  const runGames = snap.run_games_played||snap.games_played||0;
  const recent = snap.recent_games||[];
  const reachedBoss = recent.filter(g=>g.floor>=BOSS_FLOOR);
  const bossReach = recent.length ? reachedBoss.length/recent.length : 0;

  document.getElementById('summary-run-stats').innerHTML = `
    <div class="card"><div class="label">Act 1 Win Rate</div><div class="value ${wrClass(runWr)}">${(runWr*100).toFixed(1)}%</div>
      <div class="sub">${runGames} games this run</div></div>
    <div class="card"><div class="label">Boss Reach Rate</div><div class="value ${bossReach>=0.6?'good':bossReach>=0.2?'warn':'bad'}">${(bossReach*100).toFixed(0)}%</div>
      <div class="sub">last ${recent.length} games</div></div>
    <div class="card"><div class="label">Boss-Fight Win Rate</div><div class="value ${wrClass(runBfWr)}">${(runBfWr*100).toFixed(1)}%</div>
      <div class="sub">${runBfWon}/${runBfReached} boss fights this run</div></div>
  `;

  // === Section 3: Last 10 Generations ===
  const last10 = recentHistory.slice(-10);
  // Use gen_win_rate (per-generation win rate) — NOT cumulative win_rate
  const l10wr = last10.length ? last10.reduce((s,h)=>s+(h.gen_win_rate||0),0)/last10.length : 0;
  const l10br = last10.length ? last10.reduce((s,h)=>s+(h.boss_reach||0),0)/last10.length : 0;
  // For boss-fight WR over last 10 gens: compute from per-gen boss stats
  let l10bfR = 0, l10bfW = 0;
  for (let i = 1; i < last10.length; i++) {
    // Delta in boss fights between consecutive gens (within same run)
    const dr = (last10[i].boss_fights_reached||0) - (last10[i-1].boss_fights_reached||0);
    const dw = (last10[i].boss_fights_won||0) - (last10[i-1].boss_fights_won||0);
    if (dr >= 0) { l10bfR += dr; l10bfW += dw; }
    // If dr < 0, run boundary — skip that delta
  }
  const l10bfwr = l10bfR ? l10bfW / l10bfR : 0;

  document.getElementById('summary-recent-stats').innerHTML = `
    <div class="card"><div class="label">Act 1 Win Rate</div><div class="value ${wrClass(l10wr)}">${(l10wr*100).toFixed(1)}%</div>
      <div class="sub">avg of last ${last10.length} gens</div></div>
    <div class="card"><div class="label">Boss Reach Rate</div><div class="value ${l10br>=0.6?'good':l10br>=0.2?'warn':'bad'}">${(l10br*100).toFixed(0)}%</div>
      <div class="sub">avg of last ${last10.length} gens</div></div>
    <div class="card"><div class="label">Boss-Fight Win Rate</div><div class="value ${wrClass(l10bfwr)}">${(l10bfwr*100).toFixed(1)}%</div>
      <div class="sub">${l10bfW}/${l10bfR} in last ${last10.length} gens</div></div>
  `;

  // Card-pick diagnostics
  const cpAgree = snap.card_pick_agreement||0;
  const cpSpread = snap.card_pick_score_spread||0;
  const cpLoss = snap.card_pick_loss||0;
  const ooLoss = snap.other_option_loss||0;
  const cpTotal = snap.card_pick_total||0;
  const cpSkipRate = snap.card_skip_rate||0;
  const cpSkipTotal = snap.card_skip_total||0;
  const agreeClass = cpAgree >= 0.6 ? 'good' : cpAgree >= 0.3 ? 'warn' : 'bad';
  const skipClass = cpSkipRate <= 0.15 ? 'good' : cpSkipRate <= 0.4 ? 'warn' : 'bad';
  document.getElementById('summary-card-pick-stats').innerHTML = `
    <div class="card"><div class="label">Shadow Agreement</div><div class="value ${agreeClass}">${(cpAgree*100).toFixed(1)}%</div>
      <div class="sub">${cpTotal} card picks total</div></div>
    <div class="card"><div class="label">Skip Rate</div><div class="value ${skipClass}">${(cpSkipRate*100).toFixed(1)}%</div>
      <div class="sub">${cpSkipTotal} skips / ${cpTotal} offers</div></div>
    <div class="card"><div class="label">Score Spread</div><div class="value">${cpSpread.toFixed(3)}</div>
      <div class="sub">Best card vs skip (higher = more opinionated)</div></div>
    <div class="card"><div class="label">Card Pick Loss</div><div class="value">${cpLoss.toFixed(4)}</div>
      <div class="sub">vs Other Option Loss: ${ooLoss.toFixed(4)}</div></div>
  `;

  // Boss win rates — cumulative vs last 10 gens
  const body = document.getElementById('summary-boss-body');
  if (bossPerBoss && Object.keys(bossPerBoss).length) {
    const bosses = Object.entries(bossPerBoss).sort((a,b)=>b[1].total-a[1].total);
    body.innerHTML = bosses.map(([id, stats]) => {
      const cumWr = stats.total ? stats.wins/stats.total : 0;
      // Last 10 gens boss WR from recent history
      const l10bossWr = computeRecentBossWr(last10, id);
      const cumColor = wrColor(cumWr);
      const l10Color = wrColor(l10bossWr);
      const barPct = Math.min(100, cumWr*100);
      return `<tr class="boss-row">
        <td style="font-weight:bold">${humanizeBoss(id)}</td>
        <td style="color:${cumColor};font-weight:bold;text-align:right">${(cumWr*100).toFixed(1)}%</td>
        <td style="color:${l10Color};font-weight:bold;text-align:right">${l10bossWr!=null?(l10bossWr*100).toFixed(1)+'%':'-'}</td>
        <td style="text-align:left"><div class="bar" style="width:${barPct}%;max-width:120px;background:${cumColor}"></div></td>
      </tr>`;
    }).join('');
  } else {
    body.innerHTML = '<tr><td colspan="4" style="color:#484f58;text-align:center">No boss data yet</td></tr>';
  }

  // Live play stats for current version
  const rawActiveVer = activeVer === 'early' ? 'v4' : activeVer;
  const curVersionRuns = liveRuns.filter(r => r.train_version === rawActiveVer);
  const cvCompleted = curVersionRuns.filter(r => r.outcome !== 'in_progress');
  const cvWins = curVersionRuns.filter(r => r.outcome === 'win' || (r.floor||0) >= BOSS_FLOOR + 1).length;
  const cvBossReach = cvCompleted.length ? cvCompleted.filter(r=>(r.floor||0)>=BOSS_FLOOR).length/cvCompleted.length : 0;
  const cvBossFights = curVersionRuns.reduce((s,r)=>s+(r.boss_fights||[]).length, 0);
  const cvBossWins = curVersionRuns.reduce((s,r)=>s+(r.boss_fights||[]).filter(bf=>bf.outcome==='win').length, 0);
  const cvBfWr = cvBossFights ? cvBossWins/cvBossFights : 0;
  const cvWr = cvCompleted.length ? cvCompleted.filter(r=>r.outcome==='win').length/cvCompleted.length : 0;

  document.getElementById('summary-live-stats').innerHTML = `
    <div class="card"><div class="label">Live Act 1 Win Rate</div><div class="value ${wrClass(cvWr)}">${cvCompleted.length?(cvWr*100).toFixed(0)+'%':'-'}</div>
      <div class="sub">${curVersionRuns.length} runs with ${LABELS[activeVer]||activeVer}</div></div>
    <div class="card"><div class="label">Live Boss Reach</div><div class="value ${cvBossReach>=0.5?'good':cvBossReach>=0.2?'warn':'bad'}">${cvCompleted.length?(cvBossReach*100).toFixed(0)+'%':'-'}</div>
      <div class="sub">${LABELS[activeVer]||activeVer} runs</div></div>
    <div class="card"><div class="label">Live Boss Win Rate</div><div class="value ${wrClass(cvBfWr)}">${cvBossFights?(cvBfWr*100).toFixed(0)+'%':'-'}</div>
      <div class="sub">${cvBossWins}/${cvBossFights} boss fights</div></div>
  `;

  // Progress bar
  const gen = snap.generation||0;
  const total = snap.num_generations||1;
  const pct = (gen/total*100).toFixed(1);
  document.getElementById('progress-label').textContent = `Gen ${gen} / ${total} (${pct}%)`;
  document.getElementById('progress-elapsed').textContent = snap.elapsed ? `Elapsed: ${snap.elapsed}` : '';
  document.getElementById('progress-fill').style.width = pct+'%';
}

// Rough estimate of per-boss WR from recent history entries
// (history entries have boss_fight_wr but not per-boss; return null if not computable)
function computeRecentBossWr(histEntries, bossId) {
  // We don't have per-boss breakdown in history entries, so return null
  // The summary tab shows cumulative per-boss from boss_data instead
  return null;
}

function updateVersionTable(snaps, histories) {
  const body = document.getElementById('version-body');
  body.innerHTML = VERSIONS.map(v => {
    const s = snaps[v]; if (!s) return '';
    const recent = s.recent_games||[];
    const reached = recent.filter(g=>g.floor>=BOSS_FLOOR);
    const br = recent.length ? (reached.length/recent.length*100) : 0;
    let bfWr;
    if (s.boss_fight_win_rate!=null) bfWr = s.boss_fight_win_rate*100;
    else if (reached.length) bfWr = reached.filter(g=>g.outcome==='win').length/reached.length*100;
    else bfWr = null;
    const bfCell = bfWr==null ? '-' : bfWr.toFixed(0)+'%';
    const totalGens = (histories[v]||[]).length || s.generation || 0;
    return `<tr>
      <td class="${v}c">${LABELS[v]}</td>
      <td>${totalGens}</td>
      <td>${s.games_played?.toLocaleString()||'-'}</td>
      <td>${((s.win_rate||0)*100).toFixed(1)}%</td>
      <td>${br.toFixed(0)}%</td>
      <td>${bfCell}</td>
      <td>${s.card_pick_agreement!=null ? (s.card_pick_agreement*100).toFixed(0)+'%' : '-'}</td>
      <td>${s.card_pick_score_spread!=null ? s.card_pick_score_spread.toFixed(3) : '-'}</td>
      <td>${s.policy_loss?.toFixed(4)||'-'}</td>
      <td>${s.value_loss?.toFixed(4)||'-'}</td>
      <td>${s.card_pick_loss!=null ? s.card_pick_loss.toFixed(4) : '-'}</td>
      <td>${s.total_loss?.toFixed(4)||'-'}</td>
    </tr>`;
  }).join('');
}

function buildLineageDatasets(rawHistories) {
  // For each lineage, stitch all version histories end-to-end with
  // cumulative generation offsets.  Each version becomes its own dataset
  // (own color) but shares a continuous x-axis within its lineage.
  // Adjacent segments overlap by 1 point so the line connects visually.
  // Deduplicate each version's history: keep only the last entry per
  // raw generation number (handles restarts that re-train same gens).
  // Then chain versions end-to-end using max raw gen as the span.
  //
  // x-axis normalization: each lineage's total gens are normalized to
  // a 0–1 range so short lineages (V13 ~500 gens) are as visible as
  // long ones (L1 ~6800 gens). The chart x-axis shows "Training %"
  // rather than raw generation count.
  function dedup(history) {
    // Find the last restart boundary: where gen drops back to a lower
    // value (training was killed and restarted). Keep only from the
    // last restart onward so stale high-gen entries from old runs don't
    // extend the chart with outdated win rates.
    let lastRestartIdx = 0;
    for (let i = 1; i < history.length; i++) {
      if ((history[i].generation||0) < (history[i-1].generation||0)) {
        lastRestartIdx = i;
      }
    }
    const recent = history.slice(lastRestartIdx);
    const byGen = {};
    for (const pt of recent) {
      const g = pt.generation || 0;
      byGen[g] = pt;  // last write wins
    }
    return Object.values(byGen).sort((a, b) => (a.generation||0) - (b.generation||0));
  }

  const datasets = [];
  for (const lineage of LINEAGES) {
    let offset = 0;
    let segCount = 0;
    // First pass: compute total gens for this lineage (for normalization)
    let lineageTotalGens = 0;
    for (const ver of lineage.versions) {
      const raw = rawHistories[ver] || [];
      if (!raw.length) continue;
      const cleaned = dedup(raw);
      if (cleaned.length) lineageTotalGens += cleaned[cleaned.length - 1].generation || 0;
    }
    if (lineageTotalGens === 0) continue;

    for (let vi = 0; vi < lineage.versions.length; vi++) {
      const ver = lineage.versions[vi];
      const raw = rawHistories[ver] || [];
      if (!raw.length) continue;
      const cleaned = dedup(raw);
      if (!cleaned.length) continue;
      const maxRawGen = cleaned[cleaned.length - 1].generation || 0;
      const points = cleaned.map(pt => ({
        x: ((pt.generation || 0) + offset) / lineageTotalGens,
        y: pt.win_rate,
      }));
      // Bridge: connect to previous segment's last point
      const bridgePoint = segCount > 0 && datasets.length > 0
          && datasets[datasets.length - 1].data.length
        ? [datasets[datasets.length - 1].data[datasets[datasets.length - 1].data.length - 1]]
        : [];

      const thinned = thin([...bridgePoint, ...points], 500);
      const isNewest = vi === lineage.versions.length - 1;
      const bw = isNewest ? 3 : 1.8;
      const genLabel = lineageTotalGens.toLocaleString() + ' gens';
      const lbl = lineage.versions.length === 1
        ? `${LABELS[ver]} (${genLabel})`
        : `${LABELS[ver]} (${lineage.label})`;
      datasets.push({
        label: lbl,
        data: thinned,
        borderColor: COLORS[ver],
        borderWidth: bw,
        pointRadius: 0,
        tension: 0.3,
        spanGaps: true,
      });
      offset += maxRawGen;
      segCount++;
    }
  }
  return datasets;
}

function updateWinChart(rawHistories) {
  const datasets = buildLineageDatasets(rawHistories);
  winChart.data.datasets = datasets;
  winChart.options.scales.x.min = 0;
  winChart.options.scales.x.max = 1;
  winChart.options.scales.x.title.text = 'Training Progress (normalized per lineage)';
  winChart.options.scales.x.ticks.callback = function(v) { return (v*100).toFixed(0)+'%'; };
  winChart.update('none');
}

function updateGenBossWrChart(genWrData) {
  if (!genWrData || !genWrData.length) { genBossWrChart.data.datasets=[]; genBossWrChart.update('none'); return; }
  // Thin to ~150 points to reduce noise
  const thinned = thin(genWrData, 150);
  const allKeys = new Set();
  for (const pt of thinned) { for (const k of Object.keys(pt)) { if(k!=='gen') allKeys.add(k); } }
  const keyList = ['overall', ...Array.from(allKeys).filter(k=>k!=='overall').sort()];
  genBossWrChart.data.datasets = keyList.map(key => ({
    label: key==='overall' ? 'Overall' : humanizeBoss(key),
    data: thinned.filter(pt=>pt[key]!=null).map(pt=>({x:pt.gen,y:pt[key]})),
    borderColor: BOSS_CHART_COLORS[key.toUpperCase()]||BOSS_CHART_COLORS[key]||'#bc8cff',
    borderWidth: key==='overall'?3:1.8, pointRadius:0, tension:0.4,
    borderDash: key==='overall'?[]:[4,2],
  }));
  genBossWrChart.options.scales.x.max = thinned[thinned.length-1].gen||50;
  genBossWrChart.update('none');
}

function updateCardPickChart(history) {
  if (!history || !history.length) { cardPickChart.data.datasets=[]; cardPickChart.update('none'); return; }
  // Thin to ~200 points
  const thinned = thin(history, 200);
  const agreeData = thinned.filter(h=>h.card_pick_agreement!=null).map(h=>({x:h.generation,y:h.card_pick_agreement}));
  const skipData = thinned.filter(h=>h.card_skip_rate!=null).map(h=>({x:h.generation,y:h.card_skip_rate}));
  const spreadData = thinned.filter(h=>h.card_pick_score_spread!=null).map(h=>({x:h.generation,y:h.card_pick_score_spread}));
  cardPickChart.data.datasets = [
    {label:'Agreement Rate',data:agreeData,borderColor:'#58a6ff',borderWidth:2,pointRadius:0,tension:0.4,yAxisID:'yAgree'},
    {label:'Skip Rate',data:skipData,borderColor:'#f85149',borderWidth:2,pointRadius:0,tension:0.4,borderDash:[4,2],yAxisID:'yAgree'},
    {label:'Score Spread',data:spreadData,borderColor:'#d29922',borderWidth:2,pointRadius:0,tension:0.4,yAxisID:'ySpread'},
  ];
  const maxGen = thinned[thinned.length-1].generation||50;
  cardPickChart.options.scales.x.max = maxGen;
  cardPickChart.update('none');
}

function updateCardStats(cardStats) {
  const body = document.getElementById('card-stats-body');
  if (!cardStats || !cardStats.length) {
    body.innerHTML = '<tr><td colspan="7" style="text-align:center;color:#484f58">No card data yet</td></tr>';
    return;
  }
  body.innerHTML = cardStats.map(c => {
    const pickPct = (c.pick_rate*100).toFixed(1);
    const skipPct = (c.skip_rate*100).toFixed(1);
    const winPick = c.win_pick_rate!=null ? (c.win_pick_rate*100).toFixed(0)+'%' : '-';
    const barW = Math.min(120, c.pick_rate*120);
    const barColor = c.pick_rate > 0.6 ? '#3fb950' : c.pick_rate > 0.3 ? '#d29922' : '#f85149';
    return `<tr>
      <td style="text-align:left;font-family:monospace;font-size:0.85em">${c.card}</td>
      <td>${c.offered}</td>
      <td>${c.picked}</td>
      <td>${pickPct}%</td>
      <td>${skipPct}%</td>
      <td>${winPick}</td>
      <td><div class="bar" style="width:${barW}px;background:${barColor}"></div></td>
    </tr>`;
  }).join('');
}

function updateGames(games, gameBossMap) {
  const body = document.getElementById('games-body');
  body.innerHTML = games.slice().reverse().map(g => {
    const cls = g.outcome==='win' ? 'win' : 'lose';
    const pct = Math.min(100,(g.floor/BOSS_FLOOR)*100);
    const barColor = g.outcome==='win' ? '#3fb950' : '#f85149';
    const arch = g.archetype||'';
    const archColor = ARCH_COLORS[arch]||'#8b949e';
    const commit = g.commitment!=null ? `<span class="commitment-badge">${(g.commitment*100).toFixed(0)}%</span>` : '';
    const bossId = gameBossMap[String(g.num)]||'';
    const bossName = bossId ? humanizeBoss(bossId) : (g.floor>=BOSS_FLOOR?'?':'-');
    const bossColor = g.outcome==='win'?'#3fb950':(g.floor>=BOSS_FLOOR?'#f85149':'#484f58');
    return `<tr>
      <td>${g.num||''}</td><td>${g.encounter||''}</td>
      <td style="color:${bossColor};font-size:0.8em">${bossName}</td>
      <td class="${cls}">${g.outcome}</td><td>${g.floor}</td><td>${g.hp}</td>
      <td><span style="color:${archColor};text-transform:capitalize">${arch}</span>${commit}</td>
      <td style="font-size:0.8em">${g.card_picks!=null ? `${g.card_picks-g.card_skips}/${g.card_picks}` : '-'}</td>
      <td style="font-size:0.8em">${g.card_picks ? `${g.card_agrees}/${g.card_picks} (${(g.card_agrees/g.card_picks*100).toFixed(0)}%)` : '-'}</td>
      <td style="font-size:0.8em">${g.card_spread!=null ? g.card_spread.toFixed(3) : '-'}</td>
      <td><div class="bar" style="width:${pct}px;background:${barColor}"></div></td>
    </tr>`;
  }).join('');
}

function updateLiveTab(liveRuns, activeVer) {
  if (!liveRuns||!liveRuns.length) {
    document.getElementById('live-stats').innerHTML = '<div class="card" style="grid-column:1/-1;text-align:center;color:#484f58">No live play data yet</div>';
    document.getElementById('live-boss-body').innerHTML = '';
    document.getElementById('live-runs-body').innerHTML = '';
    return;
  }
  const rawActiveVer = activeVer==='early'?'v4':activeVer;
  const curRuns = liveRuns.filter(r=>r.train_version===rawActiveVer);
  const allCompleted = liveRuns.filter(r=>r.outcome!=='in_progress');

  // Resolve outcomes: in_progress (non-latest) = defeat, floor>=17 = boss fight
  const latest = liveRuns[liveRuns.length-1];
  const resolved = liveRuns.map(r => {
    let outcome = r.outcome;
    if (outcome==='in_progress' && r!==latest) outcome = 'defeat';
    // Boss defeat: died on the boss floor
    if (outcome==='defeat' && (r.floor||0) === BOSS_FLOOR) outcome = 'boss_defeat';
    if (outcome==='boss_defeat') outcome = 'boss_defeat'; // preserve from logs
    // Act 1 victory: reached beyond boss floor
    if ((outcome==='defeat'||outcome==='victory') && (r.floor||0) > BOSS_FLOOR) outcome = 'act1_victory';
    if (outcome==='win') outcome = 'act1_victory';
    return {...r, resolved_outcome: outcome};
  });

  const curResolved = resolved.filter(r=>r.train_version===rawActiveVer);
  const curCompleted = curResolved.filter(r=>r.resolved_outcome!=='in_progress');
  const curAct1Wins = curCompleted.filter(r=>r.resolved_outcome==='act1_victory').length;
  const curAct1Wr = curCompleted.length ? curAct1Wins/curCompleted.length : 0;
  const curAvgFloor = curCompleted.length ? curCompleted.reduce((s,r)=>s+(r.floor||0),0)/curCompleted.length : 0;
  const curBossReach = curCompleted.length ? curCompleted.filter(r=>(r.floor||0)>=BOSS_FLOOR).length/curCompleted.length : 0;
  const curBossFights = curRuns.reduce((s,r)=>s+(r.boss_fights||[]).length,0);
  const curBossWins = curRuns.reduce((s,r)=>s+(r.boss_fights||[]).filter(bf=>bf.outcome==='win').length,0);
  const curBfWr = curBossFights ? curBossWins/curBossFights : 0;

  document.getElementById('live-stats').innerHTML = `
    <div class="card"><div class="label">Total Runs</div><div class="ls-value" style="color:#f0f6fc;font-size:1.5em;font-weight:bold">${liveRuns.length}</div>
      <div class="sub">${curRuns.length} with ${LABELS[activeVer]||activeVer}</div></div>
    <div class="card"><div class="label">Act 1 Win Rate</div><div class="value ${wrClass(curAct1Wr)}">${curCompleted.length?(curAct1Wr*100).toFixed(0)+'%':'-'}</div>
      <div class="sub">${LABELS[activeVer]||activeVer}</div></div>
    <div class="card"><div class="label">Avg Floor</div><div class="value">${curCompleted.length?curAvgFloor.toFixed(1):'-'}</div>
      <div class="sub">${LABELS[activeVer]||activeVer}</div></div>
    <div class="card"><div class="label">Boss Reach</div><div class="value ${curBossReach>=0.5?'good':curBossReach>=0.2?'warn':'bad'}">${curCompleted.length?(curBossReach*100).toFixed(0)+'%':'-'}</div>
      <div class="sub">${LABELS[activeVer]||activeVer}</div></div>
    <div class="card"><div class="label">Boss Win Rate</div><div class="value ${wrClass(curBfWr)}">${curBossFights?(curBfWr*100).toFixed(0)+'%':'-'}</div>
      <div class="sub">${curBossWins}/${curBossFights}</div></div>
  `;

  // Per-boss WR from live runs (current version)
  const liveBossStats = {};
  for (const r of curRuns) {
    for (const bf of (r.boss_fights||[])) {
      const bid = bf.encounter_id||'Unknown';
      if (!liveBossStats[bid]) liveBossStats[bid] = {wins:0,total:0};
      liveBossStats[bid].total++;
      if (bf.outcome==='win') liveBossStats[bid].wins++;
    }
  }
  const lbb = document.getElementById('live-boss-body');
  const bosses = Object.entries(liveBossStats).sort((a,b)=>b[1].total-a[1].total);
  if (bosses.length) {
    lbb.innerHTML = bosses.map(([id,s])=>{
      const bwr=s.total?s.wins/s.total:0; const c=wrColor(bwr); const bp=Math.min(100,bwr*100);
      return `<tr class="boss-row"><td style="font-weight:bold">${humanizeBoss(id)}</td><td style="text-align:right">${s.total}</td><td style="text-align:right">${s.wins}</td><td style="color:${c};font-weight:bold;text-align:right">${(bwr*100).toFixed(1)}%</td><td style="text-align:left"><div class="bar" style="width:${bp}%;max-width:120px;background:${c}"></div></td></tr>`;
    }).join('');
  } else {
    lbb.innerHTML = '<tr><td colspan="5" style="color:#484f58;text-align:center">No boss fights yet</td></tr>';
  }

  // Run table — sort by run timestamp descending (newest first)
  const body = document.getElementById('live-runs-body');
  const sorted = resolved.slice().sort((a,b) => (b.ts||'').localeCompare(a.ts||''));
  body.innerHTML = sorted.map(r => {
    let outcomeLabel, outCls;
    if (r.resolved_outcome === 'act1_victory') { outcomeLabel='Act 1 Victory'; outCls='act1win'; }
    else if (r.resolved_outcome === 'boss_defeat') { outcomeLabel='Boss Defeat'; outCls='lose'; }
    else if (r.resolved_outcome === 'in_progress') { outcomeLabel='In Progress'; outCls=''; }
    else { outcomeLabel='Defeat'; outCls='lose'; }

    const bossInfo = (r.boss_fights||[]).map(bf=>humanizeBoss(bf.encounter_id)||'?').join(', ')||'-';
    const bossColor = r.boss_fights&&r.boss_fights.length ? (r.boss_fights.some(bf=>bf.outcome==='win')?'#3fb950':'#f85149') : '#484f58';
    const relicCount = (r.relics_gained||[]).length + (r.starting_relics||[]).length;
    const verColor = {'v9':'#39d353','v10':'#f0883e','v11':'#e05dff','v12':'#00d4aa','v13':'#ff6b9d','v14':'#79c0ff','v15':'#ffd700','v16':'#ff6347','v17':'#7b68ee'}[r.train_version]||'#8b949e';
    return `<tr onclick="showRunDetail('${r.run_id}')">
      <td style="color:${verColor};font-weight:bold;font-size:0.8em">${r.train_version||'?'}</td>
      <td>${r.config_profile||'?'}</td>
      <td style="color:#8b949e;font-size:0.75em">${r.checkpoint||'-'}</td>
      <td class="${outCls}" style="font-weight:bold">${outcomeLabel}</td>
      <td>${r.floor||0}</td>
      <td>${r.final_deck_size||r.final_deck?.length||'?'}</td>
      <td>${relicCount}</td>
      <td>${(r.combats||[]).length}</td>
      <td style="color:${bossColor};font-size:0.8em">${bossInfo}</td>
      <td style="color:#8b949e;font-size:0.75em">${formatTime(r.ts)}</td>
    </tr>`;
  }).join('');

  if (selectedRunId) {
    const run = liveRuns.find(r=>r.run_id===selectedRunId);
    if (run) renderRunDetail(run);
  }
}

let selectedRunId = null;
function showRunDetail(runId) {
  selectedRunId = selectedRunId===runId ? null : runId;
  if (window._lastLiveRuns) updateLiveTab(window._lastLiveRuns, window._lastActiveVer||'v15');
}

function renderRunDetail(run) {
  const el = document.getElementById('run-detail'); el.style.display='';
  el.innerHTML = `
    <h3>Run ${run.run_id.substring(0,8)} - ${run.outcome} at Floor ${run.floor}</h3>
    <div class="detail-grid">
      ${buildDeckPanel(run)}${buildRelicPanel(run)}${buildCombatPanel(run)}${buildBossPanel(run)}
    </div>`;
}

function buildDeckPanel(run) {
  const starting=(run.starting_deck||[]).map(c=>`<span class="card-tag">${c}</span>`).join('');
  const added=Object.entries(run.cards_added||{}).map(([c,n])=>`<span class="card-tag added">+ ${n>1?c+' x'+n:c}</span>`).join('');
  const removed=Object.entries(run.cards_removed||{}).map(([c,n])=>`<span class="card-tag removed">- ${n>1?c+' x'+n:c}</span>`).join('');
  const final=(run.final_deck||[]).map(c=>`<span class="card-tag">${c}</span>`).join('');
  return `<div class="detail-panel"><h4>Deck (${run.final_deck_size||'?'} cards)</h4>
    <div style="margin-bottom:8px"><span style="color:#8b949e;font-size:0.7em">STARTING:</span><br>${starting||'-'}</div>
    <div style="margin-bottom:8px"><span style="color:#3fb950;font-size:0.7em">ADDED:</span><br>${added||'none'}</div>
    <div style="margin-bottom:8px"><span style="color:#f85149;font-size:0.7em">REMOVED:</span><br>${removed||'none'}</div>
    <div><span style="color:#58a6ff;font-size:0.7em">FINAL:</span><br>${final||'-'}</div></div>`;
}

function buildRelicPanel(run) {
  const starters=(run.starting_relics||[]).map(r=>`<span class="relic-tag starter">${r}</span>`).join('');
  const gained=(run.relics_gained||[]).map(r=>`<span class="relic-tag">${r.name}</span>`).join('');
  const finalR=(run.final_relics||[]).map(r=>`<span class="relic-tag">${r}</span>`).join('');
  return `<div class="detail-panel"><h4>Relics (${(run.final_relics||[]).length})</h4>
    <div style="margin-bottom:8px"><span style="color:#8b949e;font-size:0.7em">STARTING:</span><br>${starters||'-'}</div>
    <div style="margin-bottom:8px"><span style="color:#d29922;font-size:0.7em">GAINED:</span><br>${gained||'none'}</div>
    <div><span style="color:#58a6ff;font-size:0.7em">FINAL:</span><br>${finalR||'-'}</div></div>`;
}

function buildCombatPanel(run) {
  const combats=run.combats||[];
  if(!combats.length) return `<div class="detail-panel"><h4>Combats</h4><span style="color:#484f58">No data</span></div>`;
  const rows=combats.map(c=>{
    const enemies=(c.enemies||[]).map(e=>`${e.name} (${e.max_hp}hp)`).join(', ');
    const outCls=c.outcome==='win'?'c-win':'c-lose';
    const hp=c.hp_before!=null&&c.hp_after!=null?`${c.hp_before} -> ${c.hp_after} HP`:'';
    const cards=(c.cards_played||[]).slice(0,12).join(', ');
    const more=(c.cards_played||[]).length>12?` +${c.cards_played.length-12} more`:'';
    const boss=c.is_boss?' <span style="color:#e05dff;font-size:0.8em">[BOSS]</span>':'';
    return `<div class="combat-row"><span class="c-floor">F${c.floor||'?'}</span> ${enemies} <span class="c-outcome ${outCls}">${c.outcome||'?'}</span>${boss} <span style="color:#8b949e;font-size:0.85em">${hp}</span><div style="color:#8b949e;font-size:0.75em">${c.turns||0} turns</div><div class="c-cards">${cards}${more}</div></div>`;
  }).join('');
  return `<div class="detail-panel"><h4>Combats (${combats.length})</h4>${rows}</div>`;
}

function buildBossPanel(run) {
  const bfs=run.boss_fights||[];
  if(!bfs.length) return `<div class="detail-panel"><h4>Boss Fights</h4><span style="color:#484f58">None</span></div>`;
  const boxes=bfs.map(bf=>{
    const enemies=(bf.enemies||[]).map(e=>`${e.name} (${e.max_hp}hp)`).join(', ');
    const oc=bf.outcome==='win'?'#3fb950':'#f85149';
    const hp=bf.hp_before!=null&&bf.hp_after!=null?`HP: ${bf.hp_before} -> ${bf.hp_after}`:'';
    return `<div class="boss-detail-box"><h4>${bf.encounter_id||'Unknown Boss'}</h4><div style="font-size:0.85em;color:${oc};font-weight:bold;margin-bottom:4px">${bf.outcome?.toUpperCase()}</div><div style="font-size:0.8em">${enemies}</div><div style="font-size:0.75em;color:#8b949e;margin-top:4px">${bf.turns} turns &middot; ${hp}</div></div>`;
  }).join('');
  return `<div class="detail-panel"><h4>Boss Fights (${bfs.length})</h4>${boxes}</div>`;
}

// ---- Main poll ----
async function poll() {
  try {
    const res = await fetch('/api/data');
    const d = await res.json();

    const earlyHist = buildEarlyHistory(d.history);
    const earlySnap = buildEarlySnap(d.snapshots, d.history);
    const histDisplay = { early: earlyHist };
    const snapsDisplay = {};
    if (earlySnap) snapsDisplay.early = earlySnap;
    for (const v of ['v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17']) {
      histDisplay[v] = d.history[v] || [];
      if (d.snapshots[v]) {
        const copy = Object.assign({}, d.snapshots[v]);
        copy.run_games_played = d.snapshots[v].games_played||0;
        copy.games_played = computeTotalGames(d.history[v]||[], d.snapshots[v].games_played||0);
        snapsDisplay[v] = copy;
      }
    }

    // Pick active version: use server's active_version if set,
    // otherwise find the newest version that has data.
    let rawActiveVer = d.active_version || '';
    if (!rawActiveVer) {
      for (const v of [...VERSIONS].reverse()) {
        if (v !== 'early' && snapsDisplay[v]) { rawActiveVer = v; break; }
      }
      if (!rawActiveVer) rawActiveVer = 'v15';
    }
    let activeVer = rawActiveVer;
    if (EARLY_SOURCES.includes(activeVer)) activeVer = 'early';
    const snap = snapsDisplay[activeVer]||snapsDisplay.v15||snapsDisplay.v14||snapsDisplay.v13||snapsDisplay.v12||snapsDisplay.v11||snapsDisplay.v10||snapsDisplay.v9||{};

    document.getElementById('active-label').innerHTML =
      `Active: <span class="${activeVer}c" style="font-weight:bold">${LABELS[activeVer]||activeVer}</span>`;

    // Boss data
    const activeBoss = (d.boss_data||{})[rawActiveVer]||{};
    const gameBossMap = activeBoss.game_boss||{};
    const liveRuns = d.live_runs||[];
    window._lastLiveRuns = liveRuns;
    window._lastActiveVer = activeVer;

    // Get recent history for active version
    const activeHistory = activeVer==='early' ? earlyHist : (d.history[rawActiveVer]||[]);

    // Summary tab
    updateSummaryTab(snap, activeVer, activeBoss.per_boss||{}, liveRuns, activeHistory);

    // Training tab
    updateVersionTable(snapsDisplay, histDisplay);
    updateWinChart(d.history);
    updateGenBossWrChart(activeBoss.gen_wr||[]);
    updateCardPickChart(activeHistory);
    updateCardStats(snap.card_stats||[]);
    updateGames(snap.recent_games||[], gameBossMap);

    // Live tab
    updateLiveTab(liveRuns, activeVer);

    document.getElementById('updated').textContent = 'Updated: ' + new Date().toLocaleTimeString();
  } catch(e) { console.error(e); }
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
                        "gen_wr": bd.get("gen_wr", []),
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
