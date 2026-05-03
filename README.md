# ClawTheSpire

An autonomous AI system for [Slay the Spire 2](https://store.steampowered.com/app/2868840/Slay_the_Spire_2/) that learns to play through self-play reinforcement learning. It combines AlphaZero-style MCTS training with an exhaustive combat solver and a deterministic strategic advisor, producing a neural network that handles the full Act 1 game loop — combat, card drafting, relic selection, map pathing, shops, events, and rest sites.

Built entirely with [Claude Code](https://claude.ai/claude-code) + the [STS2-Agent](https://github.com/CharTyr/STS2-Agent) game mod.

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                  TRAINING (Headless)                     │
│                                                         │
│  self_play.py ──► full_run.py ──► combat_engine.py      │
│  (MCTS loop)     (Act 1 sim)     (card play engine)     │
│       │                                                 │
│       ▼                                                 │
│  network.py (4-head neural net)                         │
│  ├── value_head:     win probability                    │
│  ├── policy_head:    combat card play policy            │
│  ├── option_head:    non-combat decisions               │
│  └── card_eval_head: deck-aware card pick scoring       │
│                                                         │
│  Checkpoints: alphazero_checkpoints_v17/gen_XXXX.pt     │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼ (load checkpoint)
┌─────────────────────────────────────────────────────────┐
│                 LIVE PLAY (vs real game)                 │
│                                                         │
│  runner.py (TUI game loop)                              │
│  ├── Combat: exhaustive DFS solver (solver.py)          │
│  ├── Strategy: deterministic advisor + neural net       │
│  └── Bridge: game state ↔ solver state (bridge.py)      │
│                                                         │
│  Game ◄──HTTP──► STS2-Agent mod (localhost:8081)         │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼ (metrics)
┌─────────────────────────────────────────────────────────┐
│                    MONITORING                            │
│                                                         │
│  dashboard.py (localhost:8090)                           │
│  ├── Win rates: whole version / current run / last 10   │
│  ├── Boss-fight win rates per boss (cumulative + recent)│
│  ├── Card pick preferences and agreement tracking       │
│  ├── Loss curves (policy, value, card_eval)             │
│  ├── Live play results with per-run breakdowns          │
│  └── Version comparison table (V2–V17)                  │
└─────────────────────────────────────────────────────────┘
```

## Training Pipeline

Training runs headless — no game client needed. The system simulates full Act 1 runs (17 floors: combats, elites, events, shops, rest sites, boss) and learns from outcomes via AlphaZero self-play.

Each generation: 8 games → collect samples → train network → save checkpoint. A typical training run is 12–18 hours on a MacBook Pro, producing 600–800 generations.

```bash
# Start a training run (15-hour cap, warm start from latest checkpoint)
nohup bash train-v17-15hr.sh > train-v17-15hr.log 2>&1 &
tail -f train-v17-15hr.log

# Monitor training in a browser (localhost:8090)
bash restart-dashboard.sh
```

Key training hyperparameters (V17): 500 MCTS simulations, batch size 64, 3 training epochs per generation, LR 1e-4 with cosine decay, temperature annealing from 1.0 → 0.2.

## Neural Network Architecture

Four-headed network built on card/power/relic embeddings with self-attention:

| Head | Input | Output | Purpose |
|------|-------|--------|---------|
| Value | Game state encoding (256-dim) | Scalar | Win probability estimation |
| Policy | State + action embeddings | Per-action logits | Combat card play selection |
| Option | State + option embeddings | Per-option logits | Non-combat decisions (map, shop, rest, events) |
| Card Eval | State + deck summary + relic context (357-dim) | Per-card scores | Card reward pick/skip decisions |

State encoding: 32-dim card embeddings, per-enemy projections (32-dim), power embeddings, relic embeddings (8-dim), 13-dim relic synergy features. Total vocabulary: ~580 cards, ~150 powers, ~290 relics.

## Live Play

The runner connects to the actual game via the STS2-Agent mod's HTTP API and plays autonomously using the trained network for strategic decisions and the exhaustive DFS solver for combat.

```bash
python run.py                        # Auto-play as Silent
python run.py --step                 # Step mode: press Enter per action
python run.py --dry-run              # Show decisions without executing
```

The combat solver evaluates every legal card-play ordering via depth-first search, scoring states on damage dealt, enemy threat priority, block efficiency, and power/buff values. Non-combat decisions (card rewards, map pathing, shop purchases, rest sites, events) use the trained neural network's option and card_eval heads, with a deterministic advisor as fallback.

## Dashboard

The training dashboard (`dashboard.py`) serves a live web UI at `localhost:8090` that auto-refreshes every 5 seconds:

**Summary tab** — three tiers of win rate stats (whole version aggregated across all runs, current run, last 10 generations), card selector diagnostics (shadow agreement, skip rate, score spread), per-boss win rates, and live play results.

**Training tab** — version comparison table across all training lineages (V2–V17), win rate chart by generation with lineage-based coloring, per-boss win rate timeline, card preference table (offered/picked/win-correlated), and recent game details.

**Live tab** — per-run breakdowns from real games against the actual game client, with combat details, deck evolution, and relic tracking.

## Training Version History

Each training version builds on the previous checkpoint with architectural or gameplay simulation improvements. Versions warm-start from the prior version's best checkpoint, with automatic weight migration when network dimensions change.

**V2–V3** — Foundation. 27-dim encoding, 77 encounters, reward shaping, intent-aware training signals, dynamic MCTS simulation budgets, transposition tables.

**V4–V5** — Card picking. Property-based organic card picker with archetype detection, progressive MCTS scaling (60%→140%), LR warm restarts, dynamic map pathing, improved shop/rest site logic.

**V6–V7** — Relic awareness. Removed XGBoost card picker in favor of unified organic scoring. Added `relic_synergy.py` module for relic-card synergy bonuses. Fixed relic simulation leaks (elites, treasure, boss drops, shop pool). Perceived-value heuristic for unknown relics.

**V8** — Full relic pool. ~260 Silent-relevant relics with real in-combat effects via `relic_effects.py` registry. Data-driven start-of-combat / turn-start / play-card / end-of-turn hooks. Out-of-combat pickup effects (max HP, gold, potion slots).

**V9** — Reward rebalance. Flat +1.0 for boss wins (removed HP scaling), widened win/loss gap from ~0.71 to ~1.15. Reduced potion-use penalty (0.10→0.03). Expanded potion system from 5 to ~25 types.

**V10** — Shop and event overhaul. Option types expanded (16→20), relic embed vocab expanded (13→289), shop card count fixed (3→6), dedicated event choice embeddings replacing positional placeholders.

**V11** — Enemy accuracy. 14 damage fixes, 15 pattern fixes, 21 new monsters. Silent max HP corrected (80→70). Cross-fight HP preservation bonus. Variable card reward counts. 100% relic simulation coverage (289/289).

**V12** — Simulator-live parity. 9 divergence fixes between simulator and live play (enemy intent resolution, Silent card implementations, randomized targeting, temperature alignment, bridge double-application fix). 19 new card effect implementations.

**V13** — Dedicated card evaluation head. `card_eval_head` with deck-aware mean-pooled embeddings + 2-layer MLP (336→128→64→1). Ranking loss on card picks. Shadow advisor loss. Rest site healing fix with confidence-gated guard rail.

**V14** — Card ranking refinements. Decoupled skip from card ranking loss (cards only compete with cards). Pick bonus biasing toward card acquisition in early game. Deck summary excludes base starter cards. Tuned MCTS to 500 sims, 8 games/gen.

**V15** — Real relic mechanics. ~60 Silent-eligible relics get actual combat hooks (previously proxy multipliers). Relic-aware card_eval_head expanded to 357 input dims (8 relic embed + 13 synergy features). Checkpoint migration via `pad_card_eval_weights()`. Out-of-combat relics: egg auto-upgrades, on-pickup transforms, pre-combat effects. Card pool corruption fix (`copy.copy` at card_db boundaries). ~22% Act 1 win rate, 33% boss-fight win rate.

**V16** — Economy and scoring. Empirical relic scoring from 22K V15 runs. HP preservation bonus on option samples. HP-scaled rest exploration forcing. Boosted ranking loss for win-correlated cards. Treasure chest gold rewards. Escalating card removal cost. Sim-to-live divergence fixes.

**V17** (current) — Simulator fidelity + live play combat. Act 1 map corrected to 17 floors (was incorrectly 15 in simulator). StatusCard intents now add junk cards (Dazed, Wound) to player discard pile, training the network to prioritize killing status-card enemies. Boss phase transitions: Ceremonial Beast, Vantom, and Kin Priest get more aggressive move tables below 50% HP. Combat timeout raised from 30 to 50 turns (enables poison/stall strategies). Live play: forced potion usage before MCTS loop (0 activations across 222 runs was critical gap), force-play override dropped from 50% to 1%, unknown enemy intent prediction anchored on observed damage instead of hardcoded defaults.

## Project Structure

```
ClawTheSpire/
├── sts2-solver/src/sts2_solver/
│   ├── alphazero/              # Training pipeline
│   │   ├── self_play.py        # Training loop, replay buffer, checkpoints
│   │   ├── full_run.py         # Full Act 1 run simulator (17 floors)
│   │   ├── network.py          # 4-head neural network (value/policy/option/card_eval)
│   │   ├── encoding.py         # State encoding, vocabularies, relic synergy features
│   │   └── mcts.py             # Monte Carlo Tree Search
│   ├── combat_engine.py        # Card play simulation, power ticks, relic hooks
│   ├── simulator.py            # Card pools, card rewards, event/shop handlers
│   ├── solver.py               # Exhaustive DFS combat solver
│   ├── evaluator.py            # Combat state scoring heuristics
│   ├── runner.py               # Live play game loop + TUI
│   ├── bridge.py               # Game state ↔ solver state conversion
│   ├── deterministic_advisor.py # Rule-based strategic fallback
│   ├── enemy_predict.py         # Enemy intent prediction (move tables + observed damage)
│   ├── effects.py              # Card/power effect implementations
│   ├── relic_synergy.py        # Relic-card synergy scoring
│   ├── models.py               # Card, PlayerState, CombatState dataclasses
│   ├── run_logger.py           # Event-sourced JSONL run logger
│   └── config.py               # Evaluator weights, tier lists, strategy params
├── dashboard.py                # Training dashboard (Chart.js, localhost:8090)
├── train-v17-15hr.sh           # Training launch script (current)
├── play-10-live.sh             # Run 10 live play games
├── STS2-Agent/                 # Game mod (git submodule)
└── alphazero_checkpoints_v17/  # Trained model checkpoints
```

## Prerequisites

- Python 3.11+ with [uv](https://docs.astral.sh/uv/)
- PyTorch (CPU is fine for training on Apple Silicon)
- Slay the Spire 2 + [STS2-Agent mod](https://github.com/CharTyr/STS2-Agent) (for live play only)

## License

MIT
