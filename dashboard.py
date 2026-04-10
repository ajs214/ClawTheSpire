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

# Version config: label → progress file name
VERSION_FILES = {
    "v1": "alphazero_progress.json",
    "v2": "training_v2_progress.json",
    "v3": "training_v3_progress.json",
    "v4": "training_v4_progress.json",
    "v5": "training_v5_progress.json",
    "v6": "training_v6_progress.json",
    "v7": "training_v7_progress.json",
}

# --- Shared state ---
lock = threading.Lock()
all_history: dict[str, list[dict]] = {
    "v1": [], "v2": [], "v3": [], "v4": [], "v5": [], "v6": [], "v7": [],
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
  .v5c { color: #bc8cff; }
  .v6c { color: #ff7b72; }
  .v7c { color: #ffa657; }
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
    <thead><tr><th>Version</th><th>Gens</th><th>Games</th><th>Win Rate</th><th>Boss Reach</th><th>Boss-Fight WR</th><th>Policy Loss</th><th>Value Loss</th><th>Total Loss</th></tr></thead>
    <tbody id="version-body"></tbody>
  </table>
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

<div class="recent">
  <h2>Recent Games (Active Version)</h2>
  <table>
    <thead><tr><th>#</th><th>Encounter</th><th>Outcome</th><th>Floor</th><th>HP</th><th>Archetype</th><th></th></tr></thead>
    <tbody id="games-body"></tbody>
  </table>
</div>

<div class="updated" id="updated"></div>

<script>
const COLORS = { v1: '#8b949e', v2: '#d29922', v3: '#58a6ff', v4: '#3fb950', v5: '#bc8cff', v6: '#ff7b72', v7: '#ffa657' };
const LABELS = { v1: 'V1', v2: 'V2', v3: 'V3', v4: 'V4', v5: 'V5', v6: 'V6', v7: 'V7' };
const VERSIONS = ['v1','v2','v3','v4','v5','v6','v7'];

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
    borderWidth: (v==='v7') ? 2.8 : (v==='v6') ? 2 : (v==='v5') ? 1.8 : 1.5,
    pointRadius: 0, tension: 0.3, borderDash: v==='v1' ? [4,4] : [],
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
    return `<tr>
      <td class="${v}c">${LABELS[v]}</td>
      <td>${totalGens}</td>
      <td>${s.games_played?.toLocaleString() || '—'}</td>
      <td>${((s.win_rate||0)*100).toFixed(1)}%</td>
      <td>${br.toFixed(0)}%</td>
      <td>${bfCell}</td>
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

function updateGames(games) {
  const body = document.getElementById('games-body');
  body.innerHTML = games.slice().reverse().map(g => {
    const cls = g.outcome === 'win' ? 'win' : 'lose';
    const pct = Math.min(100, (g.floor / 17) * 100);
    const barColor = g.outcome === 'win' ? '#3fb950' : '#f85149';
    const arch = g.archetype || '';
    const archColor = ARCH_COLORS[arch] || '#8b949e';
    const commit = g.commitment != null ? `<span class="commitment-badge">${(g.commitment*100).toFixed(0)}%</span>` : '';
    return `<tr>
      <td>${g.num || ''}</td>
      <td>${g.encounter || ''}</td>
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
    const activeVer = d.active_version || 'v7';
    const snap = d.snapshots[activeVer] || d.snapshots.v7 || d.snapshots.v6 || d.snapshots.v5 || {};

    updateStats(snap, activeVer);
    updateVersionTable(d.snapshots, d.history);
    updateMultiChart(policyChart, d.history, 'policy_loss');
    updateMultiChart(valueChart, d.history, 'value_loss');
    updateMultiChart(winChart, d.history, 'win_rate');
    updateMultiChart(bossChart, d.history, 'boss_reach');
    updateMultiChart(bossFightChart, d.history, 'boss_fight_wr');

    // Time chart: active version only
    const activeHist = thin(d.history[activeVer] || [], 300);
    timeChart.data.datasets[0].data = activeHist.map(h => ({ x: h.generation, y: h.gen_time }));
    timeChart.options.scales.x.type = 'linear';
    timeChart.update('none');

    updateFloors(snap.recent_games || []);
    updateGames(snap.recent_games || []);
    updateArchStats(snap);
    updateArchCharts(d.history, activeVer);
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
                payload = {
                    "snapshots": dict(snapshots),
                    "history": {k: list(v) for k, v in all_history.items()},
                    "active_version": active_version,
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
