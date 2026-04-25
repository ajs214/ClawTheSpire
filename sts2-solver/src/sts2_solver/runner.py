"""Autonomous game runner with static TUI.

Three-panel layout:
  - Top: run status bar (floor, HP, gold, screen)
  - Middle left: Solver output (last combat solution)
  - Middle right: Advisor output (last strategic decision)
  - Bottom: scrolling action log

Usage:
    python run.py                        # auto-play Ironclad from main menu
    python run.py --step                 # step mode: press Enter for each action
    python run.py --dry-run              # show decisions without executing
    python run.py --character Silent     # pick a different character
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import time
from enum import Enum
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .advisor import StrategicAdvisor
from .advisor_prompts import AUTO_ACTIONS, detect_screen_type
from .deterministic_advisor import (
    decide_boss_relic,
    decide_neow,
    decide_event_default,
    decide_card_reward,
    decide_deck_select,
    decide_map,
    decide_rest,
    decide_shop,
)
from .game_data import strip_markup
from .bridge import state_from_mcp
from .data_loader import load_cards
from .game_client import GameClient
from .game_data import load_game_data
from .run_logger import RunLogger
from .solver import solve_turn, format_solution
from .alphazero.encoding import build_vocabs_from_card_db, EncoderConfig
from .alphazero.network import STS2Network
from .alphazero.mcts import MCTS as AlphaZeroMCTS, scale_simulations
from .alphazero.state_tensor import encode_state as az_encode_state
from .alphazero.self_play import OPTION_REST, OPTION_SMITH, OPTION_EVENT_CHOICE


DEFAULT_CHARACTER = "Silent"
MAX_LOG_LINES = 50


class Runner:
    """Autonomous game runner with static TUI."""

    def __init__(
        self,
        step_mode: bool = False,
        dry_run: bool = False,
        poll_interval: float = 1.0,
        character: str = DEFAULT_CHARACTER,
        logs_dir: str | Path | None = None,
        gen: str | None = None,
    ):
        self.step_mode = step_mode
        self.dry_run = dry_run
        self.poll_interval = poll_interval
        self.character = character
        self._gen_name = gen

        self.console = Console()
        self.client = GameClient()
        self.card_db = None
        self.game_data = None
        self.advisor = None
        self.logger = RunLogger(logs_dir=Path(logs_dir) if logs_dir else None)

        # Structured event store (SQLite + Supabase + WebSocket)
        from .run_store import RunStore
        from .event_server import get_event_server
        self.store = RunStore(event_server=get_event_server())
        self._store_run_started = False

        self.game_state: dict | None = None
        self.turn_count = 0
        self.action_count = 0

        # AlphaZero MCTS (initialized lazily after card_db is loaded)
        self._mcts: AlphaZeroMCTS | None = None
        self._mcts_vocabs = None
        self._mcts_config = None
        self._card_reward_handled = False  # Reset when leaving reward screen
        self._deck_select_stuck = False  # Track stuck deck_select screens
        self._stuck_since: float | None = None  # Timestamp when we got stuck
        self._shop_visited = False  # Prevent re-opening shop after closing
        self._last_floor: int | None = None  # Track floor for shop reset
        self._last_screen_key: tuple[str, str] | None = None  # (screen, screen_type)
        self._screen_repeat_count: int = 0  # Same-screen repeat counter
        self._combat_move_indices: dict[tuple[int, str], int] = {}  # Enemy move cycle tracking

        # TUI state
        self._status_text = "[dim]Starting...[/dim]"
        self._solver_text = "[dim]Waiting for combat...[/dim]"
        self._advisor_text = "[dim]No decisions yet[/dim]"
        self._log: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._live: Live | None = None

    # ------------------------------------------------------------------
    # TUI rendering
    # ------------------------------------------------------------------

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="status", size=3),
            Layout(name="panels", ratio=3),
            Layout(name="log", ratio=2),
        )
        layout["panels"].split_row(
            Layout(name="solver"),
            Layout(name="advisor"),
        )

        # Status bar
        layout["status"].update(
            Panel(Text.from_markup(self._status_text), title="Run Status", border_style="white")
        )

        # Solver panel
        layout["solver"].update(
            Panel(Text.from_markup(self._solver_text), title="Solver", border_style="red")
        )

        # Advisor panel
        layout["advisor"].update(
            Panel(Text.from_markup(self._advisor_text), title="Advisor", border_style="blue")
        )

        # Log panel
        log_text = "\n".join(self._log) if self._log else "[dim]Waiting...[/dim]"
        layout["log"].update(
            Panel(Text.from_markup(log_text), title="Action Log", border_style="green")
        )

        return layout

    def _log_action(self, msg: str) -> None:
        self._log.append(msg)

    def _update_status(self) -> None:
        gs = self.game_state
        if gs is None:
            self._status_text = "[yellow]Waiting for game...[/yellow]"
            return

        screen = gs.get("screen", "?")
        run = gs.get("run") or {}
        floor = run.get("floor", "?")
        hp = run.get("current_hp", "?")
        max_hp = run.get("max_hp", "?")
        gold = run.get("gold", "?")

        mode = ""
        if self.dry_run:
            mode = " [yellow]\\[DRY RUN][/yellow]"
        elif self.step_mode:
            mode = " [cyan]\\[STEP][/cyan]"

        parts = [
            f"[bold]{self.character}[/bold]",
            f"Floor {floor}",
            f"HP {hp}/{max_hp}",
            f"Gold {gold}",
            f"Screen: {screen}",
            f"Turns: {self.turn_count}",
            f"Actions: {self.action_count}",
        ]
        self._status_text = " | ".join(parts) + mode

    def _refresh(self) -> None:
        if self._live:
            self._update_status()
            self._live.update(self._build_layout())

    @staticmethod
    def _is_card_reward_item(item: dict) -> bool:
        """Check if a reward item is a card reward (works with both raw and agent_view)."""
        # Raw state: reward_type = "Card"
        rtype = str(item.get("reward_type", "")).lower()
        if rtype == "card":
            return True
        # Agent view: line = "card: Add a card..."
        line = str(item.get("line", "")).lower()
        if line.startswith("card"):
            return True
        return False

    # ------------------------------------------------------------------
    # Init & main loop
    # ------------------------------------------------------------------

    def _init_deps(self) -> None:
        self.console.print("[dim]Loading card database...[/dim]")
        self.card_db = load_cards()
        self.console.print(f"[dim]Loaded {len(self.card_db)} cards[/dim]")
        self.console.print("[dim]Loading game data...[/dim]")
        self.game_data = load_game_data()
        self.advisor = StrategicAdvisor(
            self.game_data, self.client, logger=self.logger
        )

        # Initialize AlphaZero MCTS
        self.console.print("[dim]Initializing AlphaZero MCTS...[/dim]")
        self._mcts_vocabs = build_vocabs_from_card_db(self.card_db)
        self._mcts_config = EncoderConfig()
        import torch
        network = STS2Network(self._mcts_vocabs, self._mcts_config)
        # Load latest checkpoint — auto-discover the highest-numbered
        # alphazero_checkpoints_v* directory that contains a gen_*.pt file.
        # Previously this was hardcoded to prefer v3 with a v2 fallback, which
        # meant every training run from v4 onward was silently ignored by live
        # play even though the training loop was writing v4..v8..
        from pathlib import Path as _Path
        import re as _re
        _base = _Path(__file__).resolve().parents[3]

        def _version_num(p: _Path) -> int:
            m = _re.search(r"_v(\d+)$", p.name)
            return int(m.group(1)) if m else -1

        version_dirs = sorted(
            [d for d in _base.glob("alphazero_checkpoints_v*") if d.is_dir()],
            key=_version_num,
            reverse=True,
        )
        ckpt_dir = None
        ckpts: list = []
        for d in version_dirs:
            found = sorted(d.glob("gen_*.pt"), key=lambda p: p.stat().st_mtime)
            if found:
                ckpt_dir = d
                ckpts = found
                break

        self._checkpoint_name = None
        if ckpts and ckpt_dir is not None:
            ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
            saved = ckpt["model_state"]
            current = network.state_dict()
            compatible = {k: v for k, v in saved.items()
                          if k in current and v.shape == current[k].shape}
            skipped = set(saved.keys()) - set(compatible.keys())
            if any("trunk.0" in k for k in skipped):
                for k in [k for k in compatible if k.startswith("trunk.")]:
                    compatible.pop(k)
            network.load_state_dict(compatible, strict=False)
            # Include the version directory in the logged name so run metadata
            # unambiguously identifies which training generation was used.
            self._checkpoint_name = f"{ckpt_dir.name}/{ckpts[-1].name}"
            self.console.print(f"[dim]Loaded checkpoint: {self._checkpoint_name} ({len(compatible)}/{len(saved)} params)[/dim]")
        else:
            self.console.print("[dim]No checkpoint found in any alphazero_checkpoints_v* directory — using random network[/dim]")
        self._mcts = AlphaZeroMCTS(
            network, self._mcts_vocabs, self._mcts_config,
            card_db=self.card_db, device="cpu",
        )
        # Import here to avoid circular dependency — config.py is a thin router
        from .config import get_active_profile
        self.logger.metadata = {
            "advisor_model": self.advisor.model,
            "advisor_local": self.advisor.is_local,
            "checkpoint": self._checkpoint_name or "none",
            "config_profile": get_active_profile(),  # "a" (champion) or "b" (challenger)
        }
        try:
            health = self.client.get_health()
            self.logger.game_version = health.get("game_version")
            self.console.print(f"[green]Connected to game v{self.logger.game_version}[/green]")
        except ConnectionError:
            self.console.print("[yellow]Game not reachable yet — will retry[/yellow]")

    def run(self) -> None:
        self._init_deps()
        finished = False

        with Live(self._build_layout(), console=self.console, refresh_per_second=2, screen=True) as live:
            self._live = live
            try:
                while not finished:
                    finished = self._tick()
                    self._refresh()
                    if finished:
                        break
                    if self.step_mode:
                        # Drop out of Live temporarily for input
                        live.stop()
                        resp = input("[step] Enter=next, q=quit: ").strip().lower()
                        if resp == "q":
                            break
                        live.start()
                    else:
                        time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                self._log_action("[yellow]Stopped by user[/yellow]")
                self._refresh()
            finally:
                self._live = None
                self.logger.close()
                self._update_dashboard()

    def _update_dashboard(self) -> None:
        """Rebuild data.json and deploy to Vercel after a run."""
        script = Path(__file__).resolve().parents[3] / "dashboard" / "update_data.py"
        if not script.exists():
            return
        try:
            import subprocess, sys
            subprocess.run(
                [sys.executable, str(script), "--deploy"],
                timeout=60,
                capture_output=True,
            )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def _tick(self) -> bool:
        """Single iteration. Returns True if run is finished."""
        try:
            self.game_state = self.client.get_state()
        except ConnectionError:
            self._status_text = "[yellow]Waiting for game...[/yellow]"
            return False

        screen = self.game_state.get("screen", "")
        actions = self.game_state.get("available_actions", [])

        if screen == "GAME_OVER":
            self._handle_game_over()
            return True

        if screen == "MAIN_MENU":
            self._handle_main_menu(actions)
            return False

        if screen == "CHARACTER_SELECT":
            self._handle_character_select(actions)
            return False

        if screen == "MODAL":
            self._handle_modal(actions)
            return False

        if screen == "CAPSTONE_SELECTION" and "choose_capstone_option" in actions:
            self._handle_capstone_selection(actions)
            return False

        if screen == "BUNDLE_SELECTION" and "choose_bundle" in actions:
            self._handle_bundle_selection()
            return False

        # Fallback: "Choose a Pack" sub-screen sometimes reports as UNKNOWN
        # instead of BUNDLE_SELECTION.  Detect by available action.
        if "choose_bundle" in actions or "confirm_bundle" in actions:
            self._handle_bundle_selection()
            return False

        # "Choose a Pack" may also appear as an event sub-screen where
        # the only action is choose_event_option and the screen text
        # indicates pack selection.  Auto-pick option 0.
        if (screen not in ("MAIN_MENU", "CHARACTER_SELECT", "GAME_OVER")
                and "choose_event_option" in actions
                and len(actions) <= 2):
            event = self.game_state.get("event") or {}
            event_name = (event.get("name") or "").lower()
            options = event.get("options") or []
            # Detect pack/bundle sub-screens by checking option text
            _option_text = " ".join(
                (o.get("text") or o.get("label") or "") for o in options
            ).lower()
            _screen_text = (self.game_state.get("screen_text") or "").lower()
            if any(kw in f"{event_name} {_option_text} {_screen_text}"
                   for kw in ("pack", "bundle", "scroll boxes")):
                self._log_action("[dim]auto: choose_event_option(0) — pack select[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("choose_event_option", option_index=0)
                        self.action_count += 1
                        time.sleep(1.0)
                    except Exception as e:
                        self._log_action(f"  [red]Pack select failed: {e}[/red]")
                return False

        # Chest screen: open it first, then choose_treasure_relic handles relic pick
        if screen == "CHEST" and "open_chest" in actions:
            self._log_action("[dim]auto: open_chest[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("open_chest")
                    self.action_count += 1
                    time.sleep(0.5)
                except Exception as e:
                    self._log_action(f"[red]Failed to open chest: {e}[/red]")
            return False

        if not actions:
            return False

        self.logger.ensure_run(self.game_state)

        # Start a store run if we haven't yet for this run_id
        if not self._store_run_started:
            run = self.game_state.get("run") or {}
            run_id = self.game_state.get("run_id") or run.get("run_id")
            if run_id:
                self.store.start_run(
                    run_id=run_id,
                    character=run.get("character", self.character),
                    checkpoint=getattr(self, "_checkpoint_name", None),
                    gen=getattr(self, "_gen_name", None),
                    hp=run.get("current_hp"),
                    max_hp=run.get("max_hp"),
                )
                self._store_run_started = True
                self._store_run_id = run_id

        screen = self.game_state.get("screen", "")

        # Reset card reward tracking when we leave the reward screen
        if screen not in ("REWARD", "CARD_SELECTION"):
            self._card_reward_handled = False

        # Reset shop visit flag when the floor changes (not when screen changes)
        run = self.game_state.get("run") or {}
        current_floor = run.get("floor")
        if current_floor is not None and current_floor != self._last_floor:
            self._shop_visited = False
            self._last_floor = current_floor

        # Reset deck_select stuck flag when we leave the card selection screen
        if screen != "CARD_SELECTION":
            self._deck_select_stuck = False
            self._stuck_since = None

        # If stuck on a screen for too long, force end the run
        if self._deck_select_stuck and self._stuck_since:
            stuck_duration = time.monotonic() - self._stuck_since
            if stuck_duration > 60:
                self._log_action("[red]Stuck for >60s on deck select — forcing run end[/red]")
                self.logger.log_run_end(self.game_state, "stuck")
                if self._store_run_started:
                    run = self.game_state.get("run") or {}
                    self.store.end_run(
                        self._store_run_id, outcome="stuck",
                        floor=run.get("floor", 0),
                        hp=run.get("current_hp"), max_hp=run.get("max_hp"),
                    )
                    self._store_run_started = False
                return True  # Signal run is finished

        in_combat = (
            "play_card" in actions
            or ("end_turn" in actions and "COMBAT" in screen.upper())
        )

        # Track same-screen repeats to detect stuck loops
        screen_type = detect_screen_type(actions) if not in_combat else "combat"
        screen_key = (screen, screen_type)
        if screen_key == self._last_screen_key:
            self._screen_repeat_count += 1
        else:
            self._last_screen_key = screen_key
            self._screen_repeat_count = 0

        # If stuck on the same screen for too many ticks, force a default action
        if self._screen_repeat_count > 5 and not in_combat:
            self._log_action(
                f"[yellow]Stuck on {screen}/{screen_type} for {self._screen_repeat_count} ticks — forcing default[/yellow]"
            )
            self._screen_repeat_count = 0  # Reset to avoid infinite force loops
            if not self.dry_run:
                try:
                    if screen_type == "map" and "choose_map_node" in actions:
                        self._execute_with_retry("choose_map_node", option_index=0)
                    elif screen_type == "shop" and "close_shop_inventory" in actions:
                        self._execute_with_retry("close_shop_inventory")
                        self._shop_visited = True
                    else:
                        # First available action with option_index=0
                        self._execute_with_retry(actions[0], option_index=0)
                    self.action_count += 1
                except Exception as e:
                    self._log_action(f"  [red]Forced action failed: {e}[/red]")
            return False

        if in_combat:
            self._handle_combat()
        else:
            self._handle_non_combat(actions)

        return False

    # ------------------------------------------------------------------
    # Menu / character select
    # ------------------------------------------------------------------

    def _handle_main_menu(self, actions: list[str]) -> None:
        if "abandon_run" in actions:
            self._log_action("[yellow]Abandoning existing run...[/yellow]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("abandon_run")
                    time.sleep(1.0)
                    gs = self.client.get_state()
                    if "confirm_modal" in gs.get("available_actions", []):
                        self._execute_with_retry("confirm_modal")
                        time.sleep(1.0)
                except Exception as e:
                    self._log_action(f"[red]Failed to abandon run: {e}[/red]")
            return

        if "open_character_select" in actions:
            self._log_action("[cyan]Opening character select...[/cyan]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("open_character_select")
                    time.sleep(0.5)
                except Exception as e:
                    self._log_action(f"[red]Failed: {e}[/red]")

    def _handle_character_select(self, actions: list[str]) -> None:
        gs = self.game_state
        char_select = gs.get("character_select") or {}
        characters = char_select.get("characters") or []
        selected = char_select.get("selected_character_id")
        can_embark = char_select.get("can_embark", False)

        target_idx = None
        target_name = None
        for char in characters:
            name = char.get("name", "")
            char_id = char.get("character_id", "")
            if (self.character.lower() in name.lower()
                    or self.character.lower() in char_id.lower()):
                if not char.get("is_locked", False):
                    target_idx = char.get("index")
                    target_name = name
                    break

        if target_idx is None:
            available = [
                c.get("name", c.get("character_id", "?"))
                for c in characters if not c.get("is_locked", False)
            ]
            self._log_action(
                f"[red]Character '{self.character}' not found. "
                f"Available: {', '.join(available)}[/red]"
            )
            return

        needs_select = selected is None or self.character.lower() not in (selected or "").lower()
        if needs_select and "select_character" in actions:
            self._log_action(f"[cyan]Selecting {target_name}...[/cyan]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("select_character", option_index=target_idx)
                    time.sleep(0.5)
                except Exception as e:
                    self._log_action(f"[red]Failed to select: {e}[/red]")
            return

        if can_embark and "embark" in actions:
            self._log_action(f"[bold cyan]Embarking as {target_name}![/bold cyan]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("embark")
                    time.sleep(2.0)
                except Exception as e:
                    self._log_action(f"[red]Failed to embark: {e}[/red]")

    # ------------------------------------------------------------------
    # Modals
    # ------------------------------------------------------------------

    def _handle_modal(self, actions: list[str]) -> None:
        gs = self.game_state
        modal = gs.get("modal") or {}
        modal_type = modal.get("type", "")
        confirm_label = (modal.get("confirm_label") or "").lower()

        is_tutorial = any(
            kw in modal_type.lower()
            for kw in ("tutorial", "hint", "tip", "help", "learn")
        ) or any(
            kw in confirm_label
            for kw in ("tutorial", "yes", "learn", "sure")
        )

        if is_tutorial and "dismiss_modal" in actions:
            self._log_action(f"[dim]Dismissed tutorial: {modal_type}[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("dismiss_modal")
                    time.sleep(0.5)
                except Exception:
                    pass
            return

        if "dismiss_modal" in actions:
            self._log_action(f"[dim]Dismissed modal: {modal_type}[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("dismiss_modal")
                    time.sleep(0.5)
                except Exception:
                    pass
        elif "confirm_modal" in actions:
            self._log_action(f"[dim]Confirmed modal: {modal_type}[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("confirm_modal")
                    time.sleep(0.5)
                except Exception:
                    pass

    def _handle_capstone_selection(self, actions: list[str]) -> None:
        """Handle capstone/relic pack selection screens."""
        self._log_action("[dim]auto: choose_capstone_option 0[/dim]")
        if not self.dry_run:
            try:
                self._execute_with_retry("choose_capstone_option", option_index=0)
                time.sleep(1.0)
            except Exception:
                pass

    def _handle_bundle_selection(self) -> None:
        """Handle card pack/bundle selection screens (e.g. Neow's Scroll Boxes)."""
        gs = self.game_state
        actions = gs.get("available_actions", [])

        if "choose_bundle" in actions:
            self._log_action("[dim]auto: choose_bundle 0[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("choose_bundle", option_index=0)
                    time.sleep(1.0)
                except Exception:
                    pass
        elif "confirm_bundle" in actions:
            self._log_action("[dim]auto: confirm_bundle[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("confirm_bundle")
                    time.sleep(1.0)
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    # Potion categories for smart usage decisions
    _POTION_CATS: dict[str, set[str]] = {
        "heal":     {"heal", "blood", "fairy", "fruit", "regen"},
        "block":    {"block", "ghost", "shield", "iron", "armor"},
        "damage":   {"fire", "attack", "explosive", "damage", "poison",
                     "lightning", "bomb", "swift"},
        "buff":     {"strength", "flex", "dexterity", "energy", "speed",
                     "power", "stance"},
        "debuff":   {"vulnerable", "weak", "fear"},
    }

    def _classify_potion(self, name: str, desc: str) -> str | None:
        """Return the category of a potion, or None if unrecognized."""
        text = f"{name} {desc}".lower()
        for cat, keywords in self._POTION_CATS.items():
            if any(kw in text for kw in keywords):
                return cat
        return None

    def _best_damage_target(self, enemies: list[dict]) -> int | None:
        """Pick the alive enemy with the lowest HP (most likely to kill)."""
        best_idx, best_hp = None, float("inf")
        for e in enemies:
            if e.get("current_hp", 0) <= 0:
                continue
            if e.get("current_hp", 0) < best_hp:
                best_hp = e["current_hp"]
                best_idx = e.get("index", 0)
        return best_idx

    # NOTE on potion heuristics divergence (Issue 10):
    # The runner's _should_use_potion() and simulator's _use_precombat_potions() +
    # _use_emergency_potion() have slightly different logic:
    #
    # RUNNER (live play):
    #   - Before any cards: pre-check if potion should be used (this method)
    #   - During turn: within the main MCTS loop (actions.py generates use_potion
    #     actions that MCTS evaluates)
    #   - Heuristic: save potions for boss, use aggressively in boss fights
    #   - Survival thresholds: 35% HP (boss) / none explicit for non-boss
    #
    # SIMULATOR (training):
    #   - Boss fights: dump ALL offensive potions immediately at combat start
    #     (_use_precombat_potions), then heal if HP < 40%
    #   - Non-boss: only emergency heal if HP < 25% per turn
    #   - Non-boss: never dump offensive potions pre-combat
    #
    # KEY ALIGNMENT (as of this session):
    # Both now use potions during the main MCTS loop (runner has use_potion
    # actions in enumerate_actions, simulator loads potions into player state).
    # The pre-combat dump in simulator happens before MCTS runs, approximating
    # a "boss start bonus." The heuristics are similar enough for live play.

    def _should_use_potion(self, gs: dict) -> tuple[int, int | None] | None:
        """Decide whether to use a potion this turn. Returns (slot, target) or None.

        Strategy — save potions for boss fights:
        - Survival (any fight): use block/heal if we'd die, heal if HP < 35%
        - Non-boss fights: save all potions after survival checks
        - Boss fights: use potions aggressively (buff/debuff early, damage any time)
        """
        if "use_potion" not in gs.get("available_actions", []):
            return None

        run = gs.get("run") or {}
        potions = run.get("potions", [])
        combat = gs.get("combat") or {}
        player = combat.get("player") or {}
        enemies = combat.get("enemies") or []

        hp = player.get("current_hp", 0)
        max_hp = player.get("max_hp", 1)
        block = player.get("block", 0)
        turn = gs.get("turn", 0)

        # Calculate total incoming damage
        total_incoming = 0
        for e in enemies:
            if e.get("current_hp", 0) <= 0:
                continue
            for intent in e.get("intents", []):
                if intent.get("intent_type") == "Attack":
                    dmg = intent.get("damage", 0) * intent.get("hits", 1)
                    total_incoming += dmg

        unblocked = max(0, total_incoming - block)
        would_die = unblocked >= hp
        hp_pct = hp / max_hp if max_hp > 0 else 1.0

        alive_enemies = [e for e in enemies if e.get("current_hp", 0) > 0]
        total_enemy_hp = sum(e.get("current_hp", 0) for e in alive_enemies)

        # Detect boss fights via floor number (most reliable)
        from .config import STRATEGY
        floor = run.get("floor", 0)
        is_boss = floor in STRATEGY.get("boss_floors", set())

        # Fallback: very high enemy HP likely means boss
        if not is_boss and any(e.get("max_hp", 0) > 120 for e in alive_enemies):
            is_boss = True

        # Collect usable potions by category
        usable: list[tuple[int, str | None, bool]] = []  # (slot, cat, needs_target)
        for pot in potions:
            if not pot.get("occupied") or not pot.get("can_use"):
                continue
            slot = pot.get("index", 0)
            name = (pot.get("name") or "")
            desc = (pot.get("description") or "")
            cat = self._classify_potion(name, desc)
            needs_target = pot.get("requires_target", False)
            usable.append((slot, cat, needs_target))

        if not usable:
            return None

        first_alive = self._best_damage_target(enemies)

        def _target(needs_target: bool) -> int | None:
            return first_alive if needs_target else None

        # --- Priority 1: Survival — use block/heal potions if we'd die ---
        if would_die:
            for slot, cat, needs_target in usable:
                if cat == "block":
                    return (slot, _target(needs_target))
            for slot, cat, needs_target in usable:
                if cat == "heal":
                    return (slot, _target(needs_target))
            # Offense as defense — kill them before they kill us
            for slot, cat, needs_target in usable:
                if cat == "damage":
                    return (slot, _target(needs_target))

        # --- Priority 2: Heal if HP is critical ---
        if hp_pct < 0.35:
            for slot, cat, needs_target in usable:
                if cat == "heal":
                    return (slot, _target(needs_target))

        # --- Non-boss fights: save potions for the boss ---
        if not is_boss:
            return None

        # --- Boss fight: use potions aggressively ---
        # Buff/debuff potions on early turns for max value
        if turn <= 2:
            for slot, cat, needs_target in usable:
                if cat in ("buff", "debuff"):
                    return (slot, _target(needs_target))
        # Damage potions any time during boss fights
        for slot, cat, needs_target in usable:
            if cat == "damage":
                return (slot, _target(needs_target))
        # Buff potions are still good mid-fight
        for slot, cat, needs_target in usable:
            if cat == "buff":
                return (slot, _target(needs_target))
        # Debuff potions too
        for slot, cat, needs_target in usable:
            if cat == "debuff":
                return (slot, _target(needs_target))

        return None

    def _plan_full_turn(self, game_state: dict) -> list[tuple[str, int | None, int | None]]:
        """Plan a full turn's card sequence using internal MCTS simulation.

        Runs MCTS on a copy of the game state to explore the full turn's card
        sequencing without affecting the real game. Useful for understanding
        what multi-card plays are worth executing.

        Returns list of (card_name, card_hand_idx, target_idx) tuples.
        The hand_idx is relative to the current hand. An end_turn entry
        has card_name="END_TURN".
        """
        from .bridge import state_from_mcp
        from .actions import enumerate_actions as _enum_actions
        from copy import deepcopy

        try:
            sim_state = state_from_mcp(game_state, self.card_db,
                                       move_indices=self._combat_move_indices)
        except Exception:
            return []

        plan = []
        sim = deepcopy(sim_state)
        max_cards = 12

        for step_idx in range(max_cards):
            try:
                n_actions = len(_enum_actions(sim))
                sims = scale_simulations(200, n_actions)
                action, policy, root_value, _actions = self._mcts.search(
                    sim, num_simulations=sims, temperature=0.15,
                )
            except Exception:
                break

            # -- Force-play override: if MCTS wants END_TURN but there are
            # affordable playable cards, override with 1% probability.
            # Live play should be near-pure exploitation; training uses 80%
            # for exploration but live play benefits from trusting MCTS.
            if action.action_type == "end_turn":
                from .alphazero.self_play import _affordable_play_actions
                import random
                affordable = _affordable_play_actions(_enum_actions(sim), sim)
                if affordable and random.random() < 0.01:
                    action = random.choice(affordable)

            if action.action_type == "end_turn":
                plan.append(("END_TURN", None, None))
                break

            if action.action_type == "play_card":
                if action.card_idx is not None and action.card_idx < len(sim.player.hand):
                    card = sim.player.hand[action.card_idx]
                    plan.append((card.name, action.card_idx, action.target_idx))

                    # Simulate the card play internally to continue planning
                    try:
                        from .combat_engine import can_play_card, play_card, is_combat_over
                        if can_play_card(sim, action.card_idx):
                            play_card(sim, action.card_idx, action.target_idx, self.card_db)
                        else:
                            break
                        if is_combat_over(sim):
                            break
                    except Exception:
                        break
                else:
                    break
            elif action.action_type == "use_potion":
                # Record potion usage in the plan
                pot_name = "POTION"
                plan.append((pot_name, None, action.target_idx))
                # Note: We don't simulate potions in the planning loop
                # since they affect game state in ways the simulator may not model
            else:
                break

        return plan

    def _get_heuristic_fallback_action(self, sim_state, hand, enemies):
        """Fix 7: When MCTS fails, play the safest card heuristically.

        Priority:
        1. If enemies are attacking, play highest-block card
        2. If enemies have low HP, play highest-damage card
        3. Otherwise play lowest-cost playable card
        4. If nothing playable, return None (end turn)
        """
        from .combat_engine import can_play_card
        from .actions import Action, END_TURN

        if not sim_state or not hand:
            return END_TURN

        # Find all playable cards
        playable_indices = []
        for i, card in enumerate(sim_state.player.hand):
            if can_play_card(sim_state, i):
                playable_indices.append(i)

        if not playable_indices:
            return END_TURN

        # Check if enemies are attacking
        enemies_attacking = False
        min_enemy_hp = float('inf')
        for e in enemies:
            if e.get("intent_type") == "Attack":
                enemies_attacking = True
            hp = e.get("current_hp", 0)
            min_enemy_hp = min(min_enemy_hp, hp)

        # Strategy 1: If enemies are attacking, play highest-block card
        if enemies_attacking:
            best_idx = None
            best_block = -1
            for i in playable_indices:
                card = hand[i]
                block_val = card.block or 0
                if block_val > best_block:
                    best_block = block_val
                    best_idx = i
            if best_idx is not None and best_block > 0:
                return Action("play_card", card_idx=best_idx)

        # Strategy 2: If enemies have low HP, play highest-damage card
        if min_enemy_hp < float('inf') and min_enemy_hp <= 30:
            best_idx = None
            best_damage = -1
            for i in playable_indices:
                card = hand[i]
                damage_val = card.damage or 0
                if damage_val > best_damage:
                    best_damage = damage_val
                    best_idx = i
            if best_idx is not None and best_damage > 0:
                return Action("play_card", card_idx=best_idx)

        # Strategy 3: Play lowest-cost playable card
        best_idx = None
        best_cost = float('inf')
        for i in playable_indices:
            card = hand[i]
            if card.cost < best_cost:
                best_cost = card.cost
                best_idx = i

        if best_idx is not None:
            return Action("play_card", card_idx=best_idx)

        return END_TURN

    def _format_mcts_analysis(
        self, actions, policy, hand, root_value, chosen_action,
        solve_ms, total_sims, cards_played, enemy_str, turn, player,
    ) -> str:
        """Format MCTS results into a human-readable solver panel."""
        win_pct = max(0, min(100, root_value * 100))

        # Build ranked list of action options with policy weights
        options = []
        for i, (act, prob) in enumerate(zip(actions, policy)):
            if act.action_type == "end_turn":
                name = "End Turn"
            elif act.action_type == "use_potion":
                name = f"Use Potion (slot {act.potion_idx})"
            elif act.card_idx is not None and act.card_idx < len(hand):
                card = hand[act.card_idx]
                cname = f"{card.name}+" if card.upgraded else card.name
                if act.target_idx is not None:
                    name = f"{cname} → enemy {act.target_idx}"
                else:
                    name = cname
            else:
                name = str(act)
            is_chosen = (act == chosen_action)
            options.append((name, prob, is_chosen))

        # Sort by policy weight descending
        options.sort(key=lambda x: x[1], reverse=True)

        # Header
        hp = player.get("current_hp", "?")
        max_hp = player.get("max_hp", "?")
        energy = player.get("energy", "?")
        lines = [
            f"[bold]Turn {turn}[/bold] | HP {hp}/{max_hp} | Energy {energy}",
            f"vs: {enemy_str}",
            f"Win chance: [{'green' if win_pct > 50 else 'yellow' if win_pct > 20 else 'red'}]{win_pct:.0f}%[/{'green' if win_pct > 50 else 'yellow' if win_pct > 20 else 'red'}] | {total_sims} sims ({solve_ms:.0f}ms)",
            "",
        ]

        # Show top options with bar visualization
        for name, prob, is_chosen in options[:6]:
            pct = prob * 100
            bar_len = int(pct / 5)  # 20 chars = 100%
            bar = "█" * bar_len + "░" * max(0, 20 - bar_len)
            marker = " ◄" if is_chosen else ""
            if is_chosen:
                lines.append(f"[green]{bar} {pct:4.0f}% {name}{marker}[/green]")
            elif pct >= 5:
                lines.append(f"[dim]{bar} {pct:4.0f}% {name}[/dim]")
            elif pct > 0:
                lines.append(f"[dim]{'░' * 20} {pct:4.1f}% {name}[/dim]")

        # Summary of what's been played this turn
        if cards_played:
            played_str = " → ".join(cards_played)
            lines.append("")
            lines.append(f"Played: [green]{played_str}[/green]")

        return "\n".join(lines)

    def _handle_combat(self) -> None:
        gs = self.game_state
        combat = gs.get("combat") or {}
        player = combat.get("player") or {}
        enemies = combat.get("enemies") or []
        turn = gs.get("turn", "?")

        alive = [e for e in enemies if e.get("current_hp", 0) > 0]
        enemy_str = ", ".join(
            f"{e.get('name', '?')} {e.get('current_hp', '?')}hp" for e in alive
        )
        self._log_action(
            f"[red]Combat T{turn}[/red] | vs {enemy_str}"
        )

        if turn == 1 or (isinstance(turn, int) and turn <= 1):
            # Wait briefly for enemy intents to be revealed by the game
            time.sleep(0.5)
            gs = self.client.get_state()
            self.game_state = gs
            combat = gs.get("combat") or {}
            player = combat.get("player") or {}
            enemies = combat.get("enemies") or []

            self.logger.log_combat_start(gs)
            self._combat_move_indices = {}

            if self._store_run_started:
                run = gs.get("run") or {}
                self.store.log_combat_start(
                    self._store_run_id, floor=run.get("floor", 0),
                    hp=player.get("current_hp", 0), max_hp=player.get("max_hp", 0),
                    enemies=[e.get("name", "?") for e in enemies],
                )

        # ── Pre-MCTS potion forcing ──────────────────────────────
        # The heuristic _should_use_potion() is well-calibrated (saves for
        # boss, emergency heals, aggressive boss-fight usage) but MCTS
        # almost never selects use_potion over card plays.  Force the
        # heuristic's recommendation *before* entering the card-play loop
        # so potions actually get used.  Loop to drain multiple potions
        # when appropriate (e.g. boss turn 1: buff + damage potion).
        potions_used_preturn = 0
        while potions_used_preturn < 3:  # safety cap
            potion_decision = self._should_use_potion(gs)
            if potion_decision is None:
                break
            slot, target = potion_decision
            pot_name = "potion"
            potions_raw = (gs.get("run") or {}).get("potions", [])
            for p in potions_raw:
                if p.get("index") == slot:
                    pot_name = p.get("name", "potion")
                    break
            self._log_action(f"  [bold magenta]Potion: {pot_name} (slot {slot})[/bold magenta]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("use_potion", option_index=slot,
                                             target_index=target)
                    self.action_count += 1
                    potions_used_preturn += 1
                    # Re-fetch game state after potion use
                    gs = self._get_game_state()
                    self.game_state = gs
                except Exception as e:
                    self._log_action(f"  [red]Potion use failed: {e}[/red]")
                    break
            else:
                break

        # Snapshot the pre-play state for combat logging
        turn_start_gs = gs

        # Solve-one-play-re-solve loop: play one card at a time from fresh
        # game state. This accounts for relic triggers, energy generation,
        # card cost changes, and other effects the simulator doesn't model.
        from .bridge import action_to_mcp

        cards_played: list[str] = []
        targets_chosen: list[int | None] = []
        total_states = 0
        total_solve_ms = 0.0
        best_score = 0.0
        turn_root_value: float | None = None  # first MCTS value of the turn
        max_cards = 12  # safety cap to prevent infinite loops
        consecutive_rejections = 0  # track consecutive 409s (e.g. boss "ringing" mechanic)
        max_consecutive_rejections = 3  # bail after this many in a row

        # V11: Plan full turn sequence for comparison and potential direct execution
        planned_sequence = self._plan_full_turn(gs)
        plan_idx = 0
        plan_divergences = 0
        if planned_sequence:
            plan_str = " → ".join(
                p[0] for p in planned_sequence[:5]
            ) + ("..." if len(planned_sequence) > 5 else "")
            self._log_action(f"  [dim]Turn plan: {plan_str}[/dim]")

        # Update solver panel with combat state
        hand_names = ", ".join(c.get("name", "?") for c in (player.get("hand") or []))
        hp = player.get('current_hp', '?')
        max_hp = player.get('max_hp', '?')
        energy = player.get('energy', '?')
        self._solver_text = (
            f"[bold]Turn {turn}[/bold] | HP {hp}/{max_hp} | Energy {energy}\n"
            f"Hand: {hand_names}\n"
            f"vs: {enemy_str}\n\n"
            f"[dim]Thinking...[/dim]"
        )

        while len(cards_played) < max_cards:
            # Build combat state and run MCTS
            try:
                sim_state = state_from_mcp(gs, self.card_db,
                                          move_indices=self._combat_move_indices)
                hand = list(sim_state.player.hand)
                t0 = time.perf_counter()
                from .actions import enumerate_actions as _enum_actions
                _n_actions = len(_enum_actions(sim_state))
                _sims = scale_simulations(200, _n_actions)
                first_action, policy, root_value, mcts_actions = self._mcts.search(
                    sim_state, num_simulations=_sims, temperature=0.15,
                )
                solve_ms = (time.perf_counter() - t0) * 1000
                total_states += _sims
                total_solve_ms += solve_ms
                best_score = max(policy) if policy else 0
                if turn_root_value is None:
                    turn_root_value = root_value

                # -- Force-play override: if MCTS wants END_TURN but there are
                # affordable playable cards, override with 1% probability.
                # Live play should be near-pure exploitation; training uses
                # 80% for exploration but live play trusts MCTS decisions.
                if first_action.action_type == "end_turn":
                    from .alphazero.self_play import _affordable_play_actions
                    import random
                    affordable = _affordable_play_actions(_enum_actions(sim_state), sim_state)
                    if affordable and random.random() < 0.01:
                        first_action = random.choice(affordable)

                # Update solver panel with MCTS analysis
                self._solver_text = self._format_mcts_analysis(
                    mcts_actions, policy, hand, root_value, first_action,
                    solve_ms, total_states, cards_played, enemy_str, turn, player,
                )
                self._refresh()

                # V11: Compare MCTS pick to planned sequence for diagnostics
                if plan_idx < len(planned_sequence):
                    pname, _, _ = planned_sequence[plan_idx]
                    if first_action.action_type == "end_turn":
                        actual_name = "END_TURN"
                    elif first_action.action_type == "use_potion":
                        actual_name = "POTION"
                    else:
                        actual_name = (
                            hand[first_action.card_idx].name
                            if first_action.card_idx is not None and first_action.card_idx < len(hand)
                            else "?"
                        )
                    if actual_name != pname:
                        plan_divergences += 1
                    plan_idx += 1

            except Exception as e:
                self._log_action(f"[red]MCTS error: {e}[/red]")
                import traceback
                traceback.print_exc()

                # Fix 7: Heuristic fallback when MCTS fails
                # Enumerate playable cards and play the safest one
                self._log_action("[yellow]Attempting heuristic fallback...[/yellow]")
                fallback_action = self._get_heuristic_fallback_action(
                    sim_state, hand, enemies
                )
                if fallback_action:
                    first_action = fallback_action
                    self._log_action(
                        f"[yellow]Fallback: playing {fallback_action}[/yellow]"
                    )
                else:
                    self._log_action("[yellow]No playable cards; ending turn[/yellow]")
                    break

            # If MCTS says end turn, we're done
            if first_action.action_type == "end_turn":
                # Final solver panel already updated by _format_mcts_analysis above
                break

            # Handle potion usage from MCTS
            if first_action.action_type == "use_potion":
                pot_name = "potion"
                potions_raw = (gs.get("run") or {}).get("potions", [])
                for p in potions_raw:
                    if p.get("index") == first_action.potion_idx:
                        pot_name = p.get("name", "potion")
                        break
                label = f"Use {pot_name} (slot {first_action.potion_idx})"
                cards_played.append(label)
                targets_chosen.append(first_action.target_idx)

                if not self.dry_run:
                    mcp_action = action_to_mcp(first_action)
                    try:
                        self._execute_with_retry(
                            mcp_action["action"],
                            option_index=mcp_action.get("option_index"),
                            target_index=mcp_action.get("target_index"),
                        )
                        self._log_action(f"  [magenta]>[/magenta] {label}")
                        self.action_count += 1
                    except Exception as e:
                        self._log_action(f"  [red]X {label}: {e}[/red]")
                        break

                    self._refresh()
                    self._wait_for_ready(min_wait=0.5)
                    try:
                        gs = self.client.get_state()
                        self.game_state = gs
                    except Exception:
                        break

                    actions = gs.get("available_actions", [])
                    if "play_card" not in actions:
                        break
                    combat = gs.get("combat") or {}
                    player = combat.get("player") or {}
                    enemies = combat.get("enemies") or []
                continue

            # Resolve card name and target for logging
            if first_action.card_idx is not None and first_action.card_idx < len(hand):
                card = hand[first_action.card_idx]
                target_str = (
                    f" -> enemy {first_action.target_idx}"
                    if first_action.target_idx is not None else ""
                )
                logged_name = f"{card.name}+" if card.upgraded else card.name
                label = f"{logged_name}{target_str}"
                cards_played.append(logged_name)
                targets_chosen.append(first_action.target_idx)
            else:
                label = f"card_idx={first_action.card_idx}"
                cards_played.append(label)
                targets_chosen.append(None)

            if self.dry_run:
                self._log_action(f"  [dim]\\[dry-run] Would play: {label}[/dim]")
                break

            # Execute the single card play.
            # Card animations take 0.4-0.8s; give the game a brief pause
            # before sending the next action to reduce 409 rejections.
            mcp_action = action_to_mcp(first_action)
            try:
                result = self._execute_with_retry(
                    mcp_action["action"],
                    card_index=mcp_action.get("card_index"),
                    target_index=mcp_action.get("target_index"),
                )
                if not result:
                    # Card was rejected (permanent 409 — e.g. Grand Finale
                    # when draw pile isn't empty, or boss "ringing" mechanic
                    # limiting one card per turn).
                    rejected_card = label
                    if not hasattr(self, "_rejected_cards_this_turn"):
                        self._rejected_cards_this_turn = set()
                    self._rejected_cards_this_turn.add(
                        first_action.card_idx if first_action else -1
                    )
                    consecutive_rejections += 1
                    self._log_action(
                        f"  [yellow]! {rejected_card} rejected by game "
                        f"({consecutive_rejections}/{max_consecutive_rejections}) — "
                        f"re-solving without it[/yellow]"
                    )
                    # Pop the failed card from our played list so stats are accurate
                    if cards_played and cards_played[-1] == logged_name:
                        cards_played.pop()
                    if targets_chosen:
                        targets_chosen.pop()
                    # If too many consecutive rejections, assume a per-turn
                    # card limit (e.g. boss "ringing") and end the turn.
                    if consecutive_rejections >= max_consecutive_rejections:
                        self._log_action(
                            f"  [yellow]! {consecutive_rejections} consecutive "
                            f"rejections — ending turn (possible card limit)[/yellow]"
                        )
                        break
                    continue  # Re-enter the MCTS loop with fresh state
                self._log_action(f"  [green]>[/green] {label}")
                self.action_count += 1
                consecutive_rejections = 0  # reset on success
                # Clear rejected set on successful play (state has changed)
                if hasattr(self, "_rejected_cards_this_turn"):
                    self._rejected_cards_this_turn.clear()
            except Exception as e:
                self._log_action(f"  [red]X {label}: {e}[/red]")
                break

            self._refresh()

            # Wait for game to process — use a longer min_wait for combat
            # card plays (animations take longer than menu transitions).
            self._wait_for_ready(min_wait=0.5)
            try:
                gs = self.client.get_state()
                self.game_state = gs
            except Exception:
                break

            # If we left combat (enemy died, screen changed), stop
            actions = gs.get("available_actions", [])
            if "play_card" not in actions:
                break

            # Update combat locals for next iteration
            combat = gs.get("combat") or {}
            player = combat.get("player") or {}
            enemies = combat.get("enemies") or []

        # Update enemy move indices for next turn's predictions.
        # On first sight, match observed intent to move table; on subsequent
        # turns, just increment (deterministic cycling).
        from .enemy_predict import _match_move_index
        from .simulator import ENEMY_MOVE_TABLES
        for i, e_raw in enumerate(enemies):
            eid = e_raw.get("enemy_id", "")
            key = (i, eid)
            table = ENEMY_MOVE_TABLES.get(eid)
            if not table:
                continue
            if key in self._combat_move_indices:
                self._combat_move_indices[key] = (
                    (self._combat_move_indices[key] + 1) % len(table)
                )
            else:
                intents = e_raw.get("intents", [])
                it, idmg, ihits = None, None, 1
                for intent in intents:
                    itype = intent.get("intent_type", "")
                    if itype == "Attack":
                        it = "Attack"
                        idmg = intent.get("damage")
                        ihits = intent.get("hits", 1)
                    elif itype in ("Defend", "Buff", "Debuff", "StatusCard"):
                        it = it or itype
                idx = _match_move_index(eid, it, idmg, ihits)
                if idx is not None:
                    self._combat_move_indices[key] = idx

        # End turn if we're still in combat
        if not self.dry_run and "end_turn" in gs.get("available_actions", []):
            self._wait_for_ready()
            try:
                self._execute_with_retry("end_turn")
                self._log_action("  [green]>[/green] End Turn")
                self.action_count += 1
            except Exception as e:
                self._log_action(f"  [red]X End Turn: {e}[/red]")

        # V11: Log plan divergence statistics
        if planned_sequence and len(cards_played) > 0:
            div_pct = int(100 * plan_divergences / len(cards_played)) if cards_played else 0
            self._log_action(
                f"  [dim]Plan divergences: {plan_divergences}/{plan_idx} "
                f"({div_pct}%)[/dim]"
            )

        # Log the full turn (pass pre-play state for combat snapshot)
        self.logger.log_combat_turn(
            cards_played=cards_played,
            targets_chosen=targets_chosen,
            score=best_score,
            states_evaluated=total_states,
            solve_ms=total_solve_ms,
            game_state=turn_start_gs,
            network_value=turn_root_value,
            enemy_move_indices=getattr(self, '_combat_move_indices', None),
        )

        if self._store_run_started:
            run_data = (turn_start_gs.get("run") or {})
            combat_data = (turn_start_gs.get("combat") or {})
            player_data = combat_data.get("player") or {}
            self.store.log_combat_turn(
                self._store_run_id,
                floor=run_data.get("floor", 0),
                turn=turn,
                hp=player_data.get("current_hp", 0),
                max_hp=player_data.get("max_hp", 0),
                cards_played=cards_played,
                network_value=turn_root_value,
            )

        self.turn_count += 1

        # Check for combat end — only log "win" if all enemies are dead,
        # not just because the screen changed (boss phase transitions leave
        # combat temporarily for card selection screens).
        self._wait_for_ready()
        try:
            post = self.client.get_state()
            post_screen = post.get("screen", "").upper()
            if "COMBAT" not in post_screen:
                # Verify enemies are actually dead (not a mid-combat phase transition)
                combat = post.get("combat") or {}
                enemies = combat.get("enemies") or []
                all_dead = not enemies or all(
                    e.get("current_hp", 0) <= 0 for e in enemies
                )
                if all_dead:
                    self._log_action("[bold green]Combat won![/bold green]")
                    self.logger.log_combat_end(post, "win")
                    if self._store_run_started:
                        post_run = post.get("run") or {}
                        self.store.log_combat_end(
                            self._store_run_id,
                            floor=post_run.get("floor", 0),
                            hp=post_run.get("current_hp", 0),
                            max_hp=post_run.get("max_hp", 0),
                            outcome="win", turns=self.turn_count,
                        )
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Mid-combat discard helper
    # ------------------------------------------------------------------

    def _score_discard_priority(self, gs: dict, card: dict) -> int:
        """Return a discard-priority score for one card (lower = discard first).

        Priority (lowest score = discard first):
         -2  — Junk: Status/Curse cards (Wound, Slimed, Burn, Clumsy, etc.)
               Resolved via Card.is_junk on the canonical card DB, so this
               works even when the live game-state dict doesn't carry a
               'type' field (which it generally doesn't — see 807BRWNQND).
         -1  — Carry cargo: inert quest-status cards (Spoils Map, Lantern
               Key, Byrdonis Egg). Not real junk (never remove from deck),
               but in a discard-from-hand prompt they're the best target
               after real junk — dumping them from hand costs nothing
               while dumping a Strike or Sly card loses real DPS.
          1  — Unplayable this turn (unplayable_reason set or cost < 0)
               and not flagged as valuable by the tier list
          2  — Too expensive: cost > remaining energy
          3  — avoid-tier cards
          5  — Strikes / Defends (basic filler)
          8  — B-tier
         12  — A-tier
         15  — S-tier  (never discard if possible)
         16  — Protected / key cards
        """
        from .config import CHARACTER_CONFIG, detect_character
        from .deterministic_advisor import _card_tier, _resolve_card_obj

        character = detect_character(gs)
        cfg = CHARACTER_CONFIG.get(character, CHARACTER_CONFIG.get("ironclad", {}))
        protect_cards = set(cfg.get("protect_cards", [cfg.get("key_card", "Bash")]))

        combat = gs.get("combat") or {}
        player = combat.get("player") or {}
        energy = player.get("energy", 3)

        name = card.get("name", card.get("card_id", "?"))
        cost = card.get("cost", card.get("energy_cost", 99))
        if not isinstance(cost, (int, float)):
            cost = 99

        # Protected / key card — never discard.
        if name in protect_cards:
            return 16

        # Junk / carry-cargo check via the canonical Card DB. The live
        # game-state dict doesn't carry a reliable 'type' field (see
        # 807BRWNQND logs — Slimed shows up as
        # {"name":"Slimed","card_id":"SLIMED","cost":1} with no 'type'),
        # so the old in-dict card_type check never fired. Resolving
        # against the DB gives us the authoritative card_type and lets
        # Card.is_junk / Card.is_carry_cargo handle the tiering.
        card_obj = _resolve_card_obj(name)
        if card_obj is not None:
            if card_obj.is_junk:
                return -2
            if card_obj.is_carry_cargo:
                return -1

        # Fallback: if we couldn't resolve the card at all but the name
        # matches a hard-coded junk list, still flag it. This is purely
        # defensive for cases where the DB load has hiccupped; the
        # primary path is card_obj.is_junk above.
        _KNOWN_JUNK_NAMES = {
            "Wound", "Slimed", "Dazed", "Burn", "Void", "Soot",
            "Clumsy", "Doubt", "Injury", "Normality", "Regret",
            "Shame", "Writhe", "Decay", "Ascender's Bane",
            "Curse of the Bell", "Pride", "Parasite", "Necronomicurse",
        }
        if name.rstrip("+") in _KNOWN_JUNK_NAMES:
            return -2
        _KNOWN_CARRY_CARGO_NAMES = {"Spoils Map", "Lantern Key", "Byrdonis Egg"}
        if name.rstrip("+") in _KNOWN_CARRY_CARGO_NAMES:
            return -1

        # Unplayable flag. STS2's game API uses 'unplayable_reason'
        # (a string or null), while the old code checked for
        # 'unplayable' / 'is_unplayable' booleans that never exist.
        # Check both for defensiveness, plus cost == -1 which is the
        # STS2 marker for unplayable cards.
        if (card.get("unplayable_reason")
                or card.get("unplayable")
                or card.get("is_unplayable")
                or cost < 0):
            # Not junk (we already handled that), but still nothing
            # useful this turn — discard before real cards.
            return 1

        # Too expensive to play this turn.
        if cost > energy:
            return 2

        # Tier-based scoring (reverse of play priority).
        tier = _card_tier(name.rstrip("+"), character, card)
        if tier == "avoid":
            return 3
        if "Strike" in name or "Defend" in name:
            return 5
        if tier == "B":
            return 8
        if tier == "A":
            return 12
        if tier == "S":
            return 15
        return 6  # unknown tier — treat as low-B

    def _pick_card_to_discard(self, gs: dict, cards: list[dict]) -> int:
        """Pick the best single card to discard from hand.

        Delegates per-card scoring to _score_discard_priority so the
        multi-discard path can reuse the same ordering.
        """
        if not cards:
            return 0
        scored: list[tuple[int, int]] = []
        for card in cards:
            idx = card.get("index", card.get("i", 0))
            scored.append((self._score_discard_priority(gs, card), idx))
        scored.sort(key=lambda x: x[0])
        return scored[0][1]

    # ------------------------------------------------------------------
    # Non-combat
    # ------------------------------------------------------------------

    def _handle_non_combat(self, actions: list[str]) -> None:
        screen_type = detect_screen_type(actions)
        gs = self.game_state
        run = gs.get("run") or {}

        self._log_action(
            f"[blue]Floor {run.get('floor', '?')}[/blue] | {screen_type.upper()}"
        )

        # discard_potion as sole action: game is forcing a potion discard
        # (e.g. potions full after picking up a new one). Find the first
        # occupied slot to discard. Prefer slots marked can_discard, but
        # fall back to any occupied slot since the game state may not set
        # can_discard=true on this screen.
        if actions == ["discard_potion"]:
            potions = run.get("potions", [])
            discard_idx = None
            fallback_idx = None
            for p in potions:
                if p.get("occupied"):
                    if fallback_idx is None:
                        fallback_idx = p.get("index", 0)
                    if p.get("can_discard"):
                        discard_idx = p.get("index", 0)
                        break
            if discard_idx is None:
                discard_idx = fallback_idx
            if discard_idx is not None:
                pot_name = ""
                for p in potions:
                    if p.get("index") == discard_idx:
                        pot_name = f" ({p.get('name', '?')})"
                        break
                self._log_action(f"  [dim]auto: discard_potion (slot {discard_idx}{pot_name})[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("discard_potion", option_index=discard_idx)
                        self.action_count += 1
                    except Exception as e:
                        # If this slot can't be discarded, try the next occupied slot
                        self._log_action(f"  [yellow]Slot {discard_idx} not discardable, trying next[/yellow]")
                        for p in potions:
                            if p.get("occupied") and p.get("index") != discard_idx:
                                try:
                                    self._execute_with_retry("discard_potion", option_index=p["index"])
                                    self.action_count += 1
                                    break
                                except Exception:
                                    continue
            else:
                self._log_action("  [yellow]No occupied potion slots — skipping[/yellow]")
            return

        # Reward screen: collect_rewards_and_proceed auto-picks the first
        # card reward — NEVER use it when an unhandled card choice exists.
        # Instead, claim the card reward item to open the selection screen,
        # then let the advisor choose or skip.
        #
        # IMPORTANT: collect_rewards_and_proceed also auto-claims skipped card
        # rewards. After a skip, we must claim non-card rewards individually
        # first, then proceed only when no card rewards remain claimable.
        if "collect_rewards_and_proceed" in actions and screen_type != "card_reward":
            reward = gs.get("reward") or {}
            if not reward:
                reward = (gs.get("agent_view") or {}).get("reward") or {}

            # Also check agent_view reward for pending_card_choice
            agent_reward = (gs.get("agent_view") or {}).get("reward") or {}

            has_card_choice = (
                reward.get("pending_card_choice")
                or agent_reward.get("pending_card_choice")
                or "choose_reward_card" in actions
                or "skip_reward_cards" in actions
            )

            # Check reward items for card-type rewards.
            # Raw state: reward_type="Card"; agent_view: line="card: ...".
            reward_items = reward.get("rewards") or []
            if not reward_items:
                reward_items = agent_reward.get("rewards") or []
            has_card_reward_item = any(
                self._is_card_reward_item(item)
                for item in reward_items
                if item.get("claimable", True)
            )

            # If reward data is empty but we just arrived at the reward screen,
            # wait for the data to populate before auto-proceeding.
            if not reward_items and not has_card_choice and "claim_reward" in actions:
                return  # Let next tick re-check once reward data is populated

            # Debug: log reward detection state
            if "claim_reward" in actions:
                self._log_action(
                    f"  [dim]reward check: items={len(reward_items)} "
                    f"card_choice={has_card_choice} card_item={has_card_reward_item}[/dim]"
                )

            # If we already handled the card choice this reward screen,
            # claim non-card rewards individually to avoid collect_rewards_and_proceed
            # which auto-grabs the first card (even after skip_reward_cards).
            if self._card_reward_handled:
                self._card_reward_handled = False
                if "claim_reward" in actions:
                    # Claim first non-card reward item
                    for item in reward_items:
                        if not self._is_card_reward_item(item) and item.get("claimable", True):
                            idx = item.get("index", item.get("i"))
                            if idx is not None:
                                self._log_action(f"  [dim]auto: claim_reward({idx}) — non-card[/dim]")
                                if not self.dry_run:
                                    try:
                                        self._execute_with_retry("claim_reward", option_index=idx)
                                        self.action_count += 1
                                    except Exception:
                                        pass
                                return
                # No non-card rewards left, or no claim_reward action.
                # Try proceed first (doesn't auto-claim), fall back to
                # collect_rewards_and_proceed only if proceed isn't available.
                if "proceed" in actions:
                    self._log_action("  [dim]auto: proceed (post-skip)[/dim]")
                    if not self.dry_run:
                        try:
                            self._execute_with_retry("proceed")
                            self.action_count += 1
                        except Exception:
                            pass
                elif "collect_rewards_and_proceed" in actions:
                    self._log_action("  [dim]auto: collect_rewards_and_proceed (post-skip)[/dim]")
                    if not self.dry_run:
                        try:
                            self._execute_with_retry("collect_rewards_and_proceed")
                            self.action_count += 1
                        except Exception:
                            pass
                return

            if has_card_choice or has_card_reward_item:
                if "choose_reward_card" in actions or "skip_reward_cards" in actions:
                    # Card selection screen is open — let advisor handle it
                    screen_type = "card_reward"
                    # Fall through to LLM decision below
                elif has_card_reward_item and "claim_reward" in actions:
                    # Open the card selection screen by claiming the card reward
                    card_reward_idx = None
                    for item in reward_items:
                        if self._is_card_reward_item(item) and item.get("claimable", True):
                            card_reward_idx = item.get("index", item.get("i"))
                            break
                    if card_reward_idx is not None:
                        self._log_action(f"  [cyan]Opening card reward (index {card_reward_idx})...[/cyan]")
                        if not self.dry_run:
                            try:
                                self._execute_with_retry("claim_reward", option_index=card_reward_idx)
                                time.sleep(1.0)  # Wait for card data to populate
                            except Exception as e:
                                self._log_action(f"  [red]Failed to open card reward: {e}[/red]")
                    return
                else:
                    # Not ready yet — return and let next tick handle it
                    return
            else:
                # No card choice pending — safe to auto-proceed
                self._log_action("  [dim]auto: collect_rewards_and_proceed[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("collect_rewards_and_proceed")
                        self.action_count += 1
                    except Exception as e:
                        self._log_action(f"  [red]Auto-action failed: {e}[/red]")
                self.logger.log_decision(
                    game_state=gs, screen_type="auto", options=actions,
                    choice={"action": "collect_rewards_and_proceed", "option_index": None},
                    source="auto",
                )
                return

        # Auto-actions — but prioritize shop opening over proceed
        if screen_type == "auto":
            # If open_shop_inventory is available AND we haven't visited yet,
            # open the shop first (otherwise proceed would skip it entirely).
            # After visiting, _shop_visited is set so we proceed instead.
            if "open_shop_inventory" in actions and not self._shop_visited:
                action_order = ["open_shop_inventory"] + [a for a in actions if a != "open_shop_inventory"]
            else:
                action_order = actions
            for action in action_order:
                if action in AUTO_ACTIONS:
                    self._log_action(f"  [dim]auto: {action}[/dim]")
                    if not self.dry_run:
                        try:
                            self._execute_with_retry(action)
                            self.action_count += 1
                        except Exception as e:
                            self._log_action(f"  [red]Auto-action failed: {e}[/red]")
                    self.logger.log_decision(
                        game_state=gs, screen_type="auto", options=actions,
                        choice={"action": action, "option_index": None},
                        source="auto",
                    )
                    return
            return

        # Filter out side-actions that confuse the LLM
        # (discard_potion is always available but is never the primary choice)
        filtered_actions = [a for a in actions if a != "discard_potion"]
        if filtered_actions:
            gs = dict(gs)
            gs["available_actions"] = filtered_actions

        # Deck card select overlay on top of card reward screen:
        # The game can show a deck_card_select preview (e.g. card effect text)
        # while choose_reward_card is also available. If we don't dismiss the
        # overlay first, the card_reward handler gets stuck in a loop.
        if screen_type == "card_reward" and "select_deck_card" in actions:
            sel = gs.get("selection") or {}
            if sel.get("kind") == "deck_card_select":
                self._log_action("  [dim]auto: select_deck_card(0) — dismiss card preview overlay[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=0)
                    except Exception:
                        pass
                return

        # Multi-select deck screens (e.g. "Choose 2 cards to Add/Remove"):
        # select_deck_card: check if this is a real decision or an
        # informational overlay (e.g. Havoc showing "Draw 3 cards")
        if screen_type == "deck_select":
            # If we already tried and failed on this screen, skip it
            if self._deck_select_stuck:
                self._log_action("  [dim]Skipping stuck deck_select screen[/dim]")
                return

            sel = gs.get("selection") or {}
            prompt = strip_markup(sel.get("prompt") or "").lower()

            # Diagnostic: log the prompt + card count the first time we see
            # each new deck_select screen, so we can catch oddly-named
            # prompts that slip past the keyword routing below. Keyed on
            # (prompt, card_count) so repeated ticks on the same screen
            # don't spam the log but a new screen will always print.
            _card_count = len(sel.get("cards") or [])
            _diag_key = (prompt, _card_count)
            if getattr(self, "_last_deck_diag_key", None) != _diag_key:
                self._last_deck_diag_key = _diag_key
                self._log_action(
                    f"  [dim]deck_select prompt={prompt!r} cards={_card_count}[/dim]"
                )

            # "Confirm" screens (e.g. Armaments "Confirm Card to Upgrade"):
            # A card was already selected — just re-select index 0 to confirm.
            if "confirm" in prompt:
                self._log_action(f"  [dim]auto: select_deck_card(0) — confirm[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=0)
                    except Exception:
                        pass
                return

            # Mid-combat discard prompts (Survivor "Choose a card to Discard",
            # Gambler's Brew, etc.): smart discard — drop unplayable/junk first,
            # then least valuable by tier.
            _is_discard = (
                "discard" in prompt
                and "discard pile" not in prompt
            )
            if _is_discard:
                cards = sel.get("cards", [])
                pick_idx = self._pick_card_to_discard(gs, cards)
                pick_name = ""
                for c in cards:
                    if c.get("index", c.get("i", 0)) == pick_idx:
                        pick_name = c.get("name", "?")
                        break
                self._log_action(
                    f"  [dim]auto: select_deck_card({pick_idx}) — discard {pick_name}[/dim]"
                )
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=pick_idx)
                    except Exception:
                        pass
                return

            # Mid-combat card selections (Havoc "put on top of Draw Pile",
            # "Choose a card to Exhaust", etc.): pick the first non-essential
            # card quickly instead of calling the LLM.
            _COMBAT_SELECT_KW = (
                "draw pile", "exhaust", "discard pile", "put on top",
            )
            is_combat_select = any(kw in prompt for kw in _COMBAT_SELECT_KW)

            # Decision keywords that need advisor input (non-combat only).
            # NB: "smith" is the rest-site upgrade prompt in STS2 — without
            # it, rest-site upgrade pickers get misclassified as overlays
            # and dismissed with select_deck_card(0) (see IMPROVEMENTS.md).
            is_decision = not is_combat_select and any(kw in prompt for kw in (
                "choose", "remove", "upgrade", "smith",
                "transform", "add", "select",
            ))

            if is_combat_select:
                # Quick deterministic pick: avoid key card, prefer Strikes/Defends
                from .config import CHARACTER_CONFIG, detect_character
                _char = detect_character(gs)
                _key = CHARACTER_CONFIG.get(_char, {}).get("key_card", "Bash").lower()
                cards = sel.get("cards", [])
                pick_idx = 0
                for card in cards:
                    name = (card.get("name") or "").lower()
                    if _key not in name:
                        pick_idx = card.get("index", card.get("i", 0))
                        break
                self._log_action(f"  [dim]auto: select_deck_card({pick_idx}) — combat select[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=pick_idx)
                    except Exception:
                        pass
                return
            elif is_decision:
                self._handle_deck_select(gs)
            else:
                # Informational overlay — auto-select first card to dismiss.
                # If you see this log on a screen that's NOT a Havoc-style
                # overlay, the prompt needs a new keyword in is_decision
                # above.
                self._log_action(
                    f"  [yellow]auto: select_deck_card (overlay) prompt={prompt!r}[/yellow]"
                )
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=0)
                    except Exception:
                        pass
            return

        # For finished events with only a "Proceed" option, auto-handle
        if screen_type == "event" and "choose_event_option" in actions:
            event = gs.get("event") or {}
            options = event.get("options") or []
            if event.get("finished") or (
                len(options) == 1 and options[0].get("proceed")
            ):
                self._log_action("  [dim]auto: choose_event_option(0) — proceed[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("choose_event_option", option_index=0)
                        self.action_count += 1
                    except Exception as e:
                        self._log_action(f"  [red]Failed: {e}[/red]")
                return

        # For card_reward: skip if already handled (avoid re-presenting to advisor)
        if screen_type == "card_reward" and self._card_reward_handled:
            if "skip_reward_cards" in actions:
                self._log_action("  [dim]auto: skip_reward_cards (already handled)[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("skip_reward_cards")
                    except Exception:
                        pass
            return

        # For card_reward: if card options are empty, skip this tick (data not ready)
        if screen_type == "card_reward":
            reward = gs.get("reward") or {}
            if not reward:
                reward = (gs.get("agent_view") or {}).get("reward") or {}
            card_options = reward.get("card_choices") or reward.get("cards") or []
            sel = gs.get("selection") or {}
            sel_cards = sel.get("cards") or []
            if not card_options and not sel_cards:
                self._log_action("  [dim]Card reward data not ready — waiting[/dim]")
                return

        # Treasure / boss_relic: if "proceed" is available alongside the
        # relic action, it means the relic was already picked (or skipped).
        # Just proceed — don't try to pick the relic again.
        if screen_type in ("treasure", "boss_relic") and "proceed" in actions:
            self._log_action("  [dim]auto: proceed (treasure/relic done)[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("proceed")
                    self.action_count += 1
                except Exception as e:
                    self._log_action(f"  [red]proceed failed: {e}[/red]")
            return

        # General single-option auto-pick: if the screen has exactly one
        # indexed option, pick it without calling the LLM.  Applies to map,
        # event, rest, boss_relic — any screen where there's no real choice.
        _SINGLE_OPT_ACTIONS = {
            "map": "choose_map_node",
            "event": "choose_event_option",
            "rest": "choose_rest_option",
            "boss_relic": "choose_treasure_relic",
            "treasure": "choose_treasure_relic",
        }
        single_action = _SINGLE_OPT_ACTIONS.get(screen_type)
        if single_action and single_action in actions:
            # Count available options from the game state
            option_sources = {
                "map": lambda: (gs.get("map") or {}).get("available_nodes")
                    or (gs.get("map") or {}).get("nodes")
                    or ((gs.get("agent_view") or {}).get("map") or {}).get("available_nodes")
                    or ((gs.get("agent_view") or {}).get("map") or {}).get("nodes")
                    or [],
                "event": lambda: [
                    o for o in ((gs.get("event") or {}).get("options") or [])
                    if not o.get("locked")
                ],
                "rest": lambda: (gs.get("rest") or {}).get("options")
                    or ((gs.get("agent_view") or {}).get("rest") or {}).get("options")
                    or [],
                "boss_relic": lambda: (gs.get("chest") or {}).get("relics")
                    or (gs.get("reward") or {}).get("relics")
                    or ((gs.get("agent_view") or {}).get("chest") or {}).get("relics")
                    or [],
                "treasure": lambda: (gs.get("chest") or {}).get("relics")
                    or (gs.get("reward") or {}).get("relics")
                    or ((gs.get("agent_view") or {}).get("chest") or {}).get("relics")
                    or [],
            }
            opts = option_sources.get(screen_type, lambda: [])()
            if len(opts) == 1:
                idx = opts[0].get("index", opts[0].get("i", 0)) if isinstance(opts[0], dict) else 0
                self._log_action(f"  [dim]auto: {single_action}({idx}) — single option[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry(single_action, option_index=idx)
                        self.action_count += 1
                    except Exception as e:
                        self._log_action(f"  [red]Failed: {e}[/red]")
                return

        # Try network-based decisions first, fall back to deterministic.
        # USE_NETWORK_ROUTING is the A/B experimental flag:
        #   Profile A ("Self-Play")         → True  → network first on map/rest/shop
        #   Profile B ("Deterministic Base") → False → skip network on map/rest/shop
        # Card-reward and event-choice are ALWAYS routed through the
        # network regardless of profile — the learned policy for those
        # two screens is strictly better than the deterministic tier
        # lists, and training value targets were computed assuming the
        # network makes these picks, so bypassing it in live play
        # desyncs the two loops.
        from .config import USE_NETWORK_ROUTING
        _NETWORK_HANDLERS_AB = {
            "rest": self._az_decide_rest,
            "map": self._az_decide_map,
        }
        # V10: shop moves to always-on — the network now sees the same
        # option set in training and live (relics, 6 cards, real event
        # vocab). Deterministic decide_shop remains the fallback if the
        # network returns None (error / empty options).
        _NETWORK_HANDLERS_ALWAYS = {
            "card_reward": self._az_decide_card_reward,
            "shop": self._az_decide_shop,
        }
        _DETERMINISTIC_HANDLERS = {
            "rest": lambda: decide_rest(gs),
            "card_reward": lambda: decide_card_reward(gs, self.game_data),
            "map": lambda: decide_map(gs),
            "shop": lambda: decide_shop(gs, self.game_data),
            "boss_relic": lambda: decide_boss_relic(gs, self.game_data),
        }

        # Always-on network handlers (card reward).
        always_handler = _NETWORK_HANDLERS_ALWAYS.get(screen_type)
        if always_handler:
            decision = always_handler(gs)
            if decision is not None:
                self._execute_deterministic(
                    gs, decision, screen_type, actions, run,
                )
                return

        if USE_NETWORK_ROUTING:
            net_handler = _NETWORK_HANDLERS_AB.get(screen_type)
            if net_handler:
                decision = net_handler(gs)
                if decision is not None:
                    self._execute_deterministic(
                        gs, decision, screen_type, actions, run,
                    )
                    return

        handler = _DETERMINISTIC_HANDLERS.get(screen_type)
        if handler:
            decision = handler()
            self._execute_deterministic(
                gs, decision, screen_type, actions, run,
            )
            return

        # Neow event: prefer the network's option head when it returns
        # a confident match, otherwise fall back to the deterministic
        # keyword scorer. _az_decide_neow returns None on any error or
        # when no live option tag-matches a blessing, so the fall-through
        # behavior is identical to pre-network runs.
        # V10: always-on — the network has real EVENT_CHOICE_VOCAB ids
        # for Neow blessings (no longer positional placeholders).
        if screen_type == "event" and "choose_event_option" in actions:
            az_neow = self._az_decide_neow(gs)
            if az_neow is not None:
                self._execute_deterministic(
                    gs, az_neow, screen_type, actions, run,
                )
                return

            neow_decision = decide_neow(gs)
            if neow_decision is not None:
                self._execute_deterministic(
                    gs, neow_decision, screen_type, actions, run,
                )
                return

            # Non-Neow events: Tablet of Truth gets a hardcoded guard
            # first, overriding both network and heuristic. The event's
            # escalating HP cost chain and inconsistent Give Up
            # exposure make it too dangerous for either learned or
            # deterministic scorers to handle reliably — see
            # _decide_tablet_of_truth for the policy.
            tablet_decision = self._decide_tablet_of_truth(gs)
            if tablet_decision is not None:
                self._execute_deterministic(
                    gs, tablet_decision, screen_type, actions, run,
                )
                return

            # Otherwise, first try the network's option head (always
            # on, not gated by USE_NETWORK_ROUTING, so both profiles
            # use training output for event choice). If the network
            # handler bails (unreadable screen, empty options, tensor
            # failure), fall through to the deterministic sim scorer
            # which stays aligned with training value targets because
            # both call ``_evaluate_event_options``.
            az_event = self._az_decide_event_choice(gs)
            if az_event is not None:
                self._execute_deterministic(
                    gs, az_event, screen_type, actions, run,
                )
                return

            event_decision = decide_event_default(gs)
            if event_decision is not None:
                self._execute_deterministic(
                    gs, event_decision, screen_type, actions, run,
                )
                return

        # Generic / unknown screens: LLM-based decision (events no longer
        # reach this path — see decide_event_default above).
        try:
            result_str = self.advisor.advise(gs, execute=not self.dry_run)
        except Exception as e:
            self._log_action(f"[red]Advisor error: {e}[/red]")
            return

        # If the advisor recommended an invalid/failed action, fall back to a safe default
        if "not available" in result_str or "FAILED" in result_str:
            self._log_action(f"  [yellow]Invalid action — falling back[/yellow]")
            _FALLBACKS = [
                ("choose_event_option", 0),
                ("proceed", None),
                ("confirm_modal", None),
                ("dismiss_modal", None),
            ]
            for fb_action, fb_idx in _FALLBACKS:
                if fb_action not in actions:
                    continue
                if not self.dry_run:
                    try:
                        self._execute_with_retry(fb_action, option_index=fb_idx)
                    except Exception:
                        pass
                self._log_action(f"  [dim]auto: {fb_action}({fb_idx}) (fallback)[/dim]")
                self.action_count += 1
                return
            return

        # Update advisor panel
        self._advisor_text = (
            f"[bold]{screen_type.upper()}[/bold] | "
            f"Floor {run.get('floor', '?')} | "
            f"HP {run.get('current_hp', '?')}/{run.get('max_hp', '?')}\n\n"
            f"{result_str}"
        )

        lines = result_str.split("\n")
        decision_line = next((l for l in lines if l.startswith("Decision:")), lines[0] if lines else "?")
        self._log_action(f"  [blue]{decision_line}[/blue]")
        self.action_count += 1
        self._refresh()

    # ------------------------------------------------------------------
    # AlphaZero network non-combat decisions
    # ------------------------------------------------------------------

    def _az_run_state_tensors(self, gs: dict) -> tuple:
        """Build encoded state tensors from live game state for non-combat decisions.

        Uses the extended ``state_from_mcp`` (IMPROVEMENTS.md #8), which
        auto-detects non-combat screens and returns a combat-less state
        with deck loaded into ``draw_pile``. Replaces the old inline
        deck-parsing loop.

        Returns (state_tensors, hidden, hp, max_hp, gold, floor, deck_cards).
        """
        import torch
        from .bridge import state_from_mcp

        sim_state = state_from_mcp(gs, self.card_db)
        hp = sim_state.player.hp
        max_hp = sim_state.player.max_hp
        gold = sim_state.gold
        floor = sim_state.floor
        deck_cards = list(sim_state.player.draw_pile)

        st = az_encode_state(sim_state, self._mcts_vocabs, self._mcts_config)

        with torch.no_grad():
            hidden = self._mcts.network.encode_state(**st)

        return st, hidden, hp, max_hp, gold, floor, deck_cards

    def _az_decide_rest(self, gs: dict) -> "Decision | None":
        """Use network to decide rest vs upgrade at a rest site."""
        import torch
        from .deterministic_advisor import Decision

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network
            vocabs = self._mcts_vocabs

            opt_types = [OPTION_REST]
            opt_cards = [0]
            rest_idx_map = [None]  # option idx → rest option index

            # Find rest/upgrade option indices from game state
            rest_data = gs.get("rest") or (gs.get("agent_view") or {}).get("rest") or {}
            options = rest_data.get("options", [])
            game_rest_idx, game_upgrade_idx = None, None
            for i, opt in enumerate(options):
                name = (opt.get("name") or opt.get("title") or opt.get("id", "")).lower()
                idx = opt.get("index", i)
                if "rest" in name or "heal" in name or "sleep" in name:
                    game_rest_idx = idx
                elif "upgrade" in name or "smith" in name:
                    game_upgrade_idx = idx

            # Build upgrade options
            upgrade_deck_indices = []
            if game_upgrade_idx is not None:
                for di, card in enumerate(deck):
                    if not card.upgraded and card.card_type not in ("Status", "Curse"):
                        up = self.card_db.get_upgraded(card.id)
                        if up:
                            opt_types.append(OPTION_SMITH)
                            opt_cards.append(vocabs.cards.get(card.id.rstrip("+")))
                            upgrade_deck_indices.append(di)

            with torch.no_grad():
                best_idx, scores = network.pick_best_option(hidden, opt_types, opt_cards)
                nv = network.value_head(hidden).item()

            # ----------------------------------------------------------
            # Approach 1: Confidence-gated rest site guard rail
            # If HP is critically low, override the network's choice to
            # REST *unless* the network is confidently choosing SMITH
            # (score gap exceeds a margin).  As the network learns the
            # value of healing (via Approaches 2 & 3), it will develop
            # confident opinions and this guard rail self-retires.
            # ----------------------------------------------------------
            overridden = False
            hp_ratio = hp / max_hp if max_hp > 0 else 1.0
            GUARD_HP_THRESHOLD = 0.50   # only intervene when below 50% HP
            CONFIDENCE_MARGIN = 0.30    # network must beat REST by this much

            if hp_ratio < GUARD_HP_THRESHOLD and best_idx != 0 and len(scores) > 1:
                # Score gap: how much does the network prefer SMITH over REST?
                smith_score = scores[best_idx]
                rest_score = scores[0]
                gap = smith_score - rest_score

                if gap < CONFIDENCE_MARGIN:
                    # Network isn't confident enough — override to REST
                    overridden = True
                    original_idx = best_idx
                    best_idx = 0
                    self._log_action(
                        f"  [yellow]Guard rail: overriding upgrade→rest "
                        f"(HP {hp}/{max_hp}={hp_ratio:.0%}, gap={gap:.3f} < {CONFIDENCE_MARGIN})[/yellow]"
                    )

            # Build labeled scores for telemetry
            option_labels = ["Rest"]
            for di in upgrade_deck_indices:
                option_labels.append(f"Smith {deck[di].name}")
            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)} for lbl, s in zip(option_labels, scores)],
            }
            if overridden:
                hs["guard_rail_override"] = True
                hs["original_chosen"] = original_idx
                hs["hp_ratio"] = round(hp_ratio, 3)
                hs["score_gap"] = round(gap, 4)

            if best_idx == 0:
                reason = f"Network: rest (score={scores[0]:.2f})"
                if overridden:
                    reason = f"Guard rail→rest (HP={hp_ratio:.0%}, gap={gap:.3f}, orig=smith)"
                return Decision("choose_rest_option",
                                game_rest_idx if game_rest_idx is not None else 0,
                                reason,
                                network_value=nv, head_scores=hs,
                                source="network_option_head")
            else:
                card_di = upgrade_deck_indices[best_idx - 1]
                card_name = deck[card_di].name
                return Decision("choose_rest_option",
                                game_upgrade_idx if game_upgrade_idx is not None else 1,
                                f"Network: upgrade {card_name} (score={scores[best_idx]:.2f})",
                                network_value=nv, head_scores=hs,
                                source="network_option_head")
        except Exception as e:
            self._log_action(f"  [dim]Network rest failed ({e}), falling back[/dim]")
            return None

    def _az_decide_map(self, gs: dict) -> "Decision | None":
        """Use network to score map node types and pick the best."""
        import torch
        from .bridge import map_options_from_mcp
        from .deterministic_advisor import Decision

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network

            parsed = map_options_from_mcp(gs)
            opt_types = parsed["opt_types"]
            opt_cards = parsed["opt_cards"]
            actions_list = parsed["actions"]

            if not opt_types:
                return None

            with torch.no_grad():
                best_idx, scores = network.pick_best_option(hidden, opt_types, opt_cards)
                nv = network.value_head(hidden).item()

            # Build labeled scores for telemetry
            option_labels = [label for _, _, label in actions_list]
            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)} for lbl, s in zip(option_labels, scores)],
            }

            action_name, chosen_node, _label = actions_list[best_idx]
            return Decision(action_name, chosen_node,
                            f"Network: node {chosen_node} (score={scores[best_idx]:.2f})",
                            network_value=nv, head_scores=hs,
                            source="network_option_head")
        except Exception as e:
            self._log_action(f"  [dim]Network map failed ({e}), falling back[/dim]")
            return None

    def _az_decide_shop(self, gs: dict) -> "Decision | None":
        """Use network for one shop action (remove/buy/leave)."""
        import torch
        from .bridge import shop_options_from_mcp
        from .deterministic_advisor import Decision

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network

            parsed = shop_options_from_mcp(gs, deck, gold, self._mcts_vocabs)
            opt_types = parsed["opt_types"]
            opt_cards = parsed["opt_cards"]
            shop_actions = parsed["actions"]

            if not opt_types:
                return None

            with torch.no_grad():
                best_idx, scores = network.pick_best_option(hidden, opt_types, opt_cards)
                nv = network.value_head(hidden).item()

            # Build labeled scores for telemetry
            option_labels = [sa[2] for sa in shop_actions]
            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)} for lbl, s in zip(option_labels, scores)],
            }

            action_name, opt_idx, reason = shop_actions[best_idx]
            return Decision(action_name, opt_idx,
                            f"Network: {reason} (score={scores[best_idx]:.2f})",
                            network_value=nv, head_scores=hs,
                            source="network_option_head")
        except Exception as e:
            self._log_action(f"  [dim]Network shop failed ({e}), falling back[/dim]")
            return None

    def _az_decide_deck_select(self, gs: dict) -> "Decision | None":
        """Use network deck_eval_head for card removal/upgrade/transform."""
        import torch
        from .deterministic_advisor import Decision

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network
            vocabs = self._mcts_vocabs

            sel = gs.get("selection") or {}
            prompt = (sel.get("prompt") or "").lower()
            cards = sel.get("cards", [])

            if not cards:
                return None

            is_remove = "remove" in prompt or "transform" in prompt
            is_upgrade = "upgrade" in prompt

            # Build card IDs for evaluation
            card_ids = []
            card_indices = []  # game option indices
            for card_info in cards:
                card_id = card_info.get("card_id") or card_info.get("id", "")
                idx = card_info.get("index", len(card_ids))

                if is_upgrade:
                    # Score the upgraded version
                    up_id = card_id.rstrip("+") + "+"
                    card_ids.append(vocabs.cards.get(up_id.rstrip("+")))
                else:
                    card_ids.append(vocabs.cards.get(card_id.rstrip("+")))
                card_indices.append(idx)

            if not card_ids:
                return None

            with torch.no_grad():
                ids_t = torch.tensor([card_ids], dtype=torch.long)
                scores = network.evaluate_deck_change(hidden, ids_t)
                scores_list = scores[0].tolist()

            if is_remove:
                # Remove: pick lowest-scored card
                best = min(range(len(scores_list)), key=lambda i: scores_list[i])
            else:
                # Upgrade: pick highest-scored card
                best = max(range(len(scores_list)), key=lambda i: scores_list[i])

            chosen_idx = card_indices[best]
            card_name = cards[best].get("name", "?")
            nv = network.value_head(hidden).item()
            action = "remove" if is_remove else "upgrade"

            # Build labeled scores for telemetry
            option_labels = [c.get("name", "?") for c in cards[:len(scores_list)]]
            hs = {
                "head": "deck_eval",
                "chosen": best,
                "options": [{"label": lbl, "score": round(s, 4)} for lbl, s in zip(option_labels, scores_list)],
            }

            return Decision("select_deck_card", chosen_idx,
                            f"Network: {action} {card_name} (score={scores_list[best]:.2f})",
                            network_value=nv, head_scores=hs,
                            source="network_option_head")
        except Exception as e:
            self._log_action(f"  [dim]Network deck_select failed ({e}), falling back[/dim]")
            return None

    def _az_decide_card_reward(self, gs: dict) -> "Decision | None":
        """Use dedicated card_eval_head to take/skip a card reward.

        Mirrors training's ``_network_pick_card``: build
        ``[OPTION_CARD_REWARD]*N + [OPTION_CARD_SKIP]`` and let
        ``pick_best_card`` choose using deck-composition context.
        Returns a ``Decision`` that calls ``choose_reward_card(idx)``
        or ``skip_reward_cards``.

        Confidence gate: when the network's score spread across options
        is below a threshold, the card_eval_head's random-init weights
        haven't learned meaningful preferences yet — defer to the
        organic card picker heuristic.  As the head trains and develops
        a wider score spread, the gate self-retires.

        Intentionally NOT gated by ``USE_NETWORK_ROUTING`` — card
        rewards are always routed through the network in live play so
        both A and B profiles use the learned card-value policy. The
        old deterministic/tier-list path remains available as a
        fallback when the network handler returns None (bad reward
        payload, no vocabs, etc.).
        """
        import torch
        from .bridge import card_reward_options_from_mcp
        from .deterministic_advisor import Decision

        try:
            _, hidden, hp, max_hp, _gold, floor, deck_cards = (
                self._az_run_state_tensors(gs))
            network = self._mcts.network
            vocabs = self._mcts_vocabs

            parsed = card_reward_options_from_mcp(gs, vocabs)
            opt_types = parsed["opt_types"]
            opt_cards = parsed["opt_cards"]
            reward_actions = parsed["actions"]

            if not opt_types:
                return None

            # Build deck card vocab IDs for the dedicated card_eval_head
            deck_card_ids = []
            for c in deck_cards:
                base_id = c.id.rstrip("+")
                deck_card_ids.append(vocabs.cards.get(base_id, 1))  # 1=UNK

            with torch.no_grad():
                # TODO: pass relic_ids, relic_mask, and synergy_features to pick_best_card
                # for full relic-aware evaluation. For now, using backward-compatible defaults.
                best_idx, scores = network.pick_best_card(
                    hidden, deck_card_ids, opt_types, opt_cards)
                nv = network.value_head(hidden).item()

            # ----------------------------------------------------------
            # Confidence-gated card pick guard rail
            #
            # The card_eval_head starts with random weights when loaded
            # from a pre-existing checkpoint.  Until it's trained enough
            # to develop meaningful preferences, its scores will cluster
            # tightly (low spread).  When the spread is below a
            # threshold, fall back to the organic card picker which has
            # solid hand-tuned heuristics for card evaluation.
            #
            # Score spread = max(scores) - min(scores).  A well-trained
            # head will produce spreads >> 0.20 when it has opinions.
            # ----------------------------------------------------------
            CARD_CONFIDENCE_SPREAD = 0.20
            score_spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
            overridden = False
            organic_reason = ""

            if score_spread < CARD_CONFIDENCE_SPREAD:
                # Network isn't confident — ask the organic card picker
                # via the deterministic advisor which already handles
                # MCP payload parsing and has solid heuristics.
                try:
                    from .deterministic_advisor import decide_card_reward
                    organic_decision = decide_card_reward(gs, self.game_data)
                    if organic_decision is not None:
                        overridden = True
                        organic_reason = organic_decision.reasoning
                        # Map the organic decision back to our action list
                        organic_action = organic_decision.action
                        organic_opt_idx = organic_decision.option_index

                        # Find matching index in reward_actions
                        organic_best_idx = None
                        for ri, (aname, aidx, _label) in enumerate(reward_actions):
                            if aname == organic_action and aidx == organic_opt_idx:
                                organic_best_idx = ri
                                break
                        if organic_best_idx is None:
                            # Organic picked skip but we have a skip action
                            for ri, (aname, _, _) in enumerate(reward_actions):
                                if aname == organic_action:
                                    organic_best_idx = ri
                                    break

                        if organic_best_idx is not None:
                            best_idx = organic_best_idx
                            self._log_action(
                                f"  [yellow]Card guard rail: network spread={score_spread:.3f} "
                                f"< {CARD_CONFIDENCE_SPREAD}, deferring to organic picker[/yellow]"
                            )
                        else:
                            overridden = False  # couldn't map organic → our actions
                except Exception:
                    overridden = False  # organic picker failed, use network

            option_labels = [ra[2] for ra in reward_actions]
            hs = {
                "head": "card_eval",
                "chosen": best_idx,
                "deck_size": len(deck_cards),
                "score_spread": round(score_spread, 4),
                "options": [{"label": lbl, "score": round(s, 4)}
                            for lbl, s in zip(option_labels, scores)],
            }
            if overridden:
                hs["guard_rail_override"] = True
                hs["organic_reason"] = organic_reason

            action_name, opt_idx, reason = reward_actions[best_idx]
            if overridden:
                reason_str = (
                    f"Guard rail→organic: {organic_reason} "
                    f"(net spread={score_spread:.3f})"
                )
            else:
                reason_str = f"Network: {reason} (score={scores[best_idx]:.2f})"

            return Decision(
                action_name, opt_idx,
                reason_str,
                network_value=nv, head_scores=hs,
                source="network_card_eval_head" if not overridden else "organic_guard_rail")
        except Exception as e:
            self._log_action(
                f"  [dim]Network card_reward failed ({e}), falling back[/dim]")
            return None

    def _az_decide_event_choice(self, gs: dict) -> "Decision | None":
        """Use network option-head to pick a non-Neow event option.

        Mirrors training's event-choice branch in ``full_run``: builds
        ``[OPTION_EVENT_CHOICE]*N`` with real EVENT_CHOICE_VOCAB IDs
        via the dedicated event_choice_embed table (V10).

        Intentionally NOT gated by ``USE_NETWORK_ROUTING`` — non-Neow
        events always route through the network in live play so the
        training value targets (which were computed against the
        network's pick) match what live play actually does. The
        deterministic ``decide_event_default`` path remains the fallback
        when the network handler returns None (locked options,
        unreadable screen, tensor build failure, etc.).
        """
        import torch
        from .bridge import event_options_from_mcp
        from .deterministic_advisor import Decision

        try:
            event = gs.get("event") or (
                gs.get("agent_view") or {}).get("event") or {}
            if not event:
                return None
            # Skip Neow — _az_decide_neow handles that screen.
            event_name = (
                event.get("name") or event.get("event_id") or "").lower()
            run = gs.get("run") or {}
            floor = int(run.get("floor") or 0)
            if "neow" in event_name or floor <= 1:
                return None

            _, hidden, _hp, _max_hp, _gold, _floor, _deck = (
                self._az_run_state_tensors(gs))
            network = self._mcts.network

            parsed = event_options_from_mcp(gs, self._mcts_vocabs)
            opt_types = parsed["opt_types"]
            opt_cards = parsed["opt_cards"]
            event_actions = parsed["actions"]

            if not opt_types:
                return None

            with torch.no_grad():
                best_idx, scores = network.pick_best_option(
                    hidden, opt_types, opt_cards)
                nv = network.value_head(hidden).item()

            option_labels = [ea[2] for ea in event_actions]
            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)}
                            for lbl, s in zip(option_labels, scores)],
            }

            action_name, opt_idx, reason = event_actions[best_idx]
            display_name = (
                event.get("name") or event.get("event_id") or "event")
            return Decision(
                action_name, opt_idx,
                f"Network {display_name}: {reason} "
                f"(score={scores[best_idx]:.2f})",
                network_value=nv, head_scores=hs,
                source="network_option_head")
        except Exception as e:
            self._log_action(
                f"  [dim]Network event_choice failed ({e}), falling back[/dim]")
            return None

    def _decide_tablet_of_truth(self, gs: dict) -> "Decision | None":
        """Hardcoded guard for the Tablet of Truth event.

        The event chains 4+ decipher pages that each cost -3 max HP and
        grant a random upgrade, ending with a "Lose Everything" page
        that upgrades all cards at the cost of nearly all remaining HP.
        In practice the escalating HP cost bankrupts the run long
        before the full-deck upgrade pays off, and the continuation
        pages don't consistently expose a "Give Up" choice — so the
        only reliable control we have is on the INITIAL page.

        Strategy:
          * INITIAL page (where a "smash" option is available): always
            pick Smash (+20 HP). Never start the decipher chain — it's
            the only binding decision we get, so the once-per-run cap
            requested by the user is best enforced as "zero decipher
            chains".
          * Continuation page (no smash option): look for any label
            that matches "give up"/"leave"/"stop"/"decline" and pick
            it. Otherwise return None and let the network/heuristic
            handle it — we're locked in and have no override.

        Returns a Decision with ``source="hardcoded_tablet_guard"`` so
        the log line is distinguishable from the network and the
        heuristic fallback.
        """
        from .deterministic_advisor import Decision

        event = gs.get("event") or (gs.get("agent_view") or {}).get("event") or {}
        if not event:
            return None
        event_id = (
            event.get("event_id") or event.get("id")
            or event.get("name") or "").upper().replace(" ", "_")
        if "TABLET_OF_TRUTH" not in event_id and "TABLET OF TRUTH" not in (
                event.get("name") or "").upper():
            return None

        raw_options = event.get("options") or []
        labelled: list[tuple[int, str, int]] = []
        for i, opt in enumerate(raw_options):
            if opt.get("locked"):
                continue
            label = (
                opt.get("title") or opt.get("name")
                or (opt.get("description") or "")[:40] or ""
            ).lower()
            game_idx = opt.get("index", i)
            labelled.append((i, label, int(game_idx)))

        if not labelled:
            return None

        # INITIAL page → always Smash. Don't start the chain.
        for _, label, game_idx in labelled:
            if "smash" in label:
                return Decision(
                    "choose_event_option", game_idx,
                    "Tablet of Truth: Smash (+20 HP, skipping decipher chain)",
                    source="hardcoded_tablet_guard",
                )

        # Continuation page → look for an escape route.
        _ESCAPE_KEYWORDS = ("give up", "leave", "stop", "decline", "refuse")
        for _, label, game_idx in labelled:
            if any(kw in label for kw in _ESCAPE_KEYWORDS):
                return Decision(
                    "choose_event_option", game_idx,
                    f"Tablet of Truth: escape via '{label}'",
                    source="hardcoded_tablet_guard",
                )

        # Locked-in continuation with no escape — nothing we can do.
        # Fall through to the normal handlers so at least something
        # gets clicked and the event progresses.
        return None

    def _az_decide_neow(self, gs: dict) -> "Decision | None":
        """Use network to pick a Neow blessing on floor 1.

        Training path: ``play_full_run`` scores the canonical
        ``NEOW_BLESSINGS`` list in its static order, with
        ``opt_types == [OPTION_EVENT_CHOICE] * N`` and
        ``opt_cards == [vocab_id_for_each_blessing]`` (V10 real IDs).
        We mirror that exactly here.

        The network returns an index into ``NEOW_BLESSINGS`` (training's
        synthetic list). We then map that back to the live game's option
        index by tag-matching: classify each live option's text with
        ``classify_neow_option_text``, then pick the live option whose
        tag matches the network's chosen blessing. If no live option
        tag-matches (e.g. modded Neow or follow-up sub-screen), return
        None so the caller falls through to ``decide_neow``.
        """
        import torch
        from .deterministic_advisor import Decision
        from .simulator import (
            NEOW_BLESSINGS, _NEOW_KEY_TO_TAG,
            classify_neow_option_text,
        )

        try:
            # Detect that this really is Neow. Use the same signal as
            # decide_neow: floor <= 1 AND either "neow" in event name or
            # at least one option with a Neow-ish keyword. Bail otherwise.
            event = gs.get("event") or (gs.get("agent_view") or {}).get("event") or {}
            live_options = event.get("options") or []
            if not live_options:
                return None
            event_name = (event.get("name") or event.get("event_id") or "").lower()
            run = gs.get("run") or {}
            floor = int(run.get("floor") or 0)
            if "neow" not in event_name and floor > 1:
                return None

            # Classify every live option up front. Drop anything that
            # doesn't classify as a Neow tag so follow-up sub-screens
            # (e.g. "Scroll Boxes" pack picker) fall through.
            live_tagged: list[tuple[int, str, str]] = []  # (game_idx, tag, label)
            for i, opt in enumerate(live_options):
                if opt.get("locked"):
                    continue
                name = opt.get("name") or opt.get("title") or ""
                desc = opt.get("description") or opt.get("desc") or ""
                text = f"{name} — {desc}"
                tag = classify_neow_option_text(text)
                if tag == "unknown":
                    continue
                game_idx = opt.get("index", i)
                live_tagged.append((game_idx, tag, name or text.strip()))
            if not live_tagged:
                return None

            # Build the network query against the static NEOW_BLESSINGS
            # list — the same tensor shape the option head was trained on.
            _, hidden, _hp, _max_hp, _gold, _floor, _deck = self._az_run_state_tensors(gs)
            network = self._mcts.network
            vocabs = self._mcts_vocabs

            n_bless = len(NEOW_BLESSINGS)
            opt_types = [OPTION_EVENT_CHOICE] * n_bless
            # V10: real EVENT_CHOICE_VOCAB ids via dedicated embedding
            from .simulator import _event_choice_vocab_id, _NEOW_EVENT_ID as _NEI
            opt_cards = [_event_choice_vocab_id(_NEI, i) for i in range(n_bless)]

            with torch.no_grad():
                best_idx, scores = network.pick_best_option(
                    hidden, opt_types, opt_cards)
                nv = network.value_head(hidden).item()

            # Map network pick (index into NEOW_BLESSINGS) back to a
            # live option. Walk blessings in descending network-score
            # order and take the first live option whose tag matches.
            order = sorted(range(n_bless), key=lambda i: scores[i], reverse=True)
            best_bless = None
            matched_live_idx = None
            matched_live_label = None
            for b_idx in order:
                key = NEOW_BLESSINGS[b_idx][0]
                tag = _NEOW_KEY_TO_TAG.get(key, "unknown")
                for game_idx, live_tag, label in live_tagged:
                    if live_tag == tag:
                        best_bless = b_idx
                        matched_live_idx = game_idx
                        matched_live_label = label
                        break
                if matched_live_idx is not None:
                    break

            if matched_live_idx is None:
                # No blessing tag matched any live option. Let
                # decide_neow handle it — it runs the same scorer
                # against the live option texts directly, which is more
                # robust for modded/unusual Neow variants.
                return None

            # Telemetry labels (training blessings, not live options, so
            # the report shows what the network actually scored).
            option_labels = [f"{NEOW_BLESSINGS[i][0]} [{NEOW_BLESSINGS[i][1]}]"
                             for i in range(n_bless)]
            hs = {
                "head": "option_eval",
                "chosen": best_bless,
                "options": [{"label": lbl, "score": round(s, 4)}
                            for lbl, s in zip(option_labels, scores)],
            }

            return Decision(
                "choose_event_option", matched_live_idx,
                f"Network Neow: {matched_live_label} "
                f"(blessing={NEOW_BLESSINGS[best_bless][0]}, "
                f"score={scores[best_bless]:.2f})",
                network_value=nv, head_scores=hs,
                source="network_option_head",
            )
        except Exception as e:
            self._log_action(f"  [dim]Network Neow failed ({e}), falling back[/dim]")
            return None

    # ------------------------------------------------------------------
    # Deterministic decision execution
    # ------------------------------------------------------------------

    def _execute_deterministic(
        self,
        gs: dict,
        decision,  # deterministic_advisor.Decision
        screen_type: str,
        actions: list[str],
        run: dict,
    ) -> None:
        """Execute a deterministic advisor decision."""
        from .deterministic_advisor import Decision

        # Validate action is available
        if decision.action not in actions:
            self._log_action(
                f"  [yellow]Deterministic action '{decision.action}' not available, "
                f"falling back[/yellow]"
            )
            # Fallback by screen type
            _FALLBACKS = {
                "rest": ("choose_rest_option", 0),
                "card_reward": ("skip_reward_cards", None),
                "map": ("choose_map_node", 0),
                "shop": ("close_shop_inventory", None),
                "boss_relic": ("choose_treasure_relic", 0),
                "treasure": ("choose_treasure_relic", 0),
                "deck_select": ("select_deck_card", 0),
            }
            fb = _FALLBACKS.get(screen_type)
            if fb and fb[0] in actions:
                decision = Decision(fb[0], fb[1], "fallback",
                                    source="fallback_first_action")
            else:
                return

        # Log the decision
        self._log_action(
            f"  [blue]Decision: {decision.action}"
            f"{f' ({decision.option_index})' if decision.option_index is not None else ''}"
            f" — {decision.reasoning}[/blue]"
        )

        if self.logger:
            self.logger.log_decision(
                game_state=gs,
                screen_type=screen_type,
                options=actions,
                choice={
                    "action": decision.action,
                    "option_index": decision.option_index,
                    "reasoning": decision.reasoning,
                },
                source=decision.source,
                network_value=decision.network_value,
                head_scores=decision.head_scores,
            )

        if self._store_run_started:
            self.store.log_decision(
                self._store_run_id,
                floor=run.get("floor", 0),
                hp=run.get("current_hp", 0),
                max_hp=run.get("max_hp", 0),
                screen_type=screen_type,
                choice=decision.reasoning,
                network_value=decision.network_value,
                head_scores=decision.head_scores,
            )

        # Execute
        if not self.dry_run:
            try:
                self._execute_with_retry(
                    decision.action, option_index=decision.option_index,
                )
            except Exception as e:
                self._log_action(f"  [red]Execution failed: {e}[/red]")
                return

        # Post-execution bookkeeping
        if screen_type == "card_reward":
            self._card_reward_handled = True
        if screen_type == "shop" and decision.action == "close_shop_inventory":
            self._shop_visited = True

        # Update advisor panel
        self._advisor_text = (
            f"[bold]{screen_type.upper()}[/bold] | "
            f"Floor {run.get('floor', '?')} | "
            f"HP {run.get('current_hp', '?')}/{run.get('max_hp', '?')}\n\n"
            f"[green]\\[deterministic][/green] {decision.action}"
            f"{f' (idx={decision.option_index})' if decision.option_index is not None else ''}\n"
            f"{decision.reasoning}"
        )

        self.action_count += 1
        self._refresh()

    # ------------------------------------------------------------------
    # Multi-select deck screens
    # ------------------------------------------------------------------

    def _handle_deck_select(self, gs: dict) -> None:
        """Handle deck card selection screens (add, remove, upgrade, transform).

        For single-select (upgrade, transform): use the advisor.
        For multi-select (e.g. "Choose 2 to Remove"): pick deterministically
        using Strikes first, then Defends, never key card — since the advisor
        would give the same answer every call and multi-select toggling is
        unreliable via the API.
        """
        sel = gs.get("selection") or {}
        prompt_text = strip_markup(sel.get("prompt") or "").lower()
        cards = sel.get("cards", [])

        # Detect multi-select from prompt (e.g. "Choose 2 cards to Remove")
        import re
        multi_match = re.search(r"choose\s+(\d+)", prompt_text)
        is_multi = multi_match is not None and int(multi_match.group(1)) > 1

        if is_multi:
            self._handle_multi_deck_select(gs, cards, prompt_text)
        else:
            self._handle_single_deck_select(gs)

    def _handle_single_deck_select(self, gs: dict) -> None:
        """Single-select deck screen — deterministic tier-list decision.

        NOTE: _az_decide_deck_select is disabled because the network's
        evaluate_deck_change method was never implemented.  Calling it
        adds ~200ms of wasted tensor work + exception overhead per screen.
        Re-enable once evaluate_deck_change exists on STS2Network.
        """
        decision = decide_deck_select(gs)
        actions = gs.get("available_actions", [])
        run = gs.get("run") or {}
        self._execute_deterministic(gs, decision, "deck_select", actions, run)

        # Wait for screen to change or confirm
        time.sleep(0.5)
        try:
            gs = self.client.get_state()
        except Exception:
            return
        self.game_state = gs
        if "confirm_selection" in gs.get("available_actions", []):
            if not self.dry_run:
                try:
                    self._execute_with_retry("confirm_selection")
                except Exception:
                    pass
            self._log_action("  [dim]auto: confirm_selection[/dim]")

    def _handle_multi_deck_select(self, gs: dict, cards: list, prompt_text: str) -> None:
        """Multi-select deck screen — pick deterministically.

        For remove: Strikes first, then Defends, never key card.
        For other multi-selects: pick sequentially from index 0.
        """
        import re
        multi_match = re.search(r"choose\s+(\d+)", prompt_text)
        num_to_pick = int(multi_match.group(1)) if multi_match else 2

        is_remove = "remove" in prompt_text
        is_upgrade = "upgrade" in prompt_text
        # Multi-discard surfaces: Gambling Chip ("Discard any number of
        # cards"), multi-card Survivor variants, etc. The older
        # _handle_multi_deck_select had no is_discard branch, so these
        # prompts fell into the 'else' path below and picked cards in
        # raw hand-index order — totally ignoring Wounds. Route them
        # through the same priority scorer the single-discard path uses
        # so junk gets queued first.
        is_discard = (
            "discard" in prompt_text
            and "discard pile" not in prompt_text
        )

        # Build priority order for indices
        if is_discard:
            scored = []
            for card in cards:
                idx = card.get("index", 0)
                score = self._score_discard_priority(gs, card)
                scored.append((score, idx))
            scored.sort(key=lambda x: x[0])  # lowest score = discard first
            priority = [idx for _, idx in scored]
            self._log_action(
                f"  [cyan]Multi-discard: priority {priority[:num_to_pick]} "
                f"(junk-first)[/cyan]"
            )
        elif is_remove:
            # Remove Strikes first, then Defends, then others, NEVER key card
            from .config import CHARACTER_CONFIG, detect_character
            _char = detect_character(gs)
            _key = CHARACTER_CONFIG.get(_char, {}).get("key_card", "Bash").lower()
            priority = []
            for card in cards:
                name = (card.get("name") or "").lower()
                idx = card.get("index", 0)
                if _key in name:
                    continue  # Never remove key card
                if "strike" in name:
                    priority.insert(0, idx)  # Strikes first
                elif "defend" in name:
                    priority.append(idx)  # Defends after Strikes
                else:
                    priority.append(idx)  # Others last
            self._log_action(f"  [cyan]Multi-remove: picking {num_to_pick} from priority {priority[:num_to_pick]}[/cyan]")
        else:
            # For upgrade/other: just pick sequentially
            priority = [card.get("index", i) for i, card in enumerate(cards)]

        picked = 0
        attempted_indices: set[int] = set()
        max_attempts = num_to_pick * 3  # Safety limit

        for attempt in range(max_attempts):
            if picked >= num_to_pick:
                break

            # Walk the priority list for the first index we haven't
            # already tried this round. Using attempt-as-index was
            # fragile when a previous iteration refreshed state and
            # `picked` advanced mid-loop.
            idx: int | None = None
            for cand in priority:
                if cand not in attempted_indices:
                    idx = cand
                    break
            if idx is None:
                break
            attempted_indices.add(idx)

            card_name = next(
                (c.get("name", "?") for c in cards if c.get("index") == idx),
                f"index {idx}",
            )
            self._log_action(f"  [cyan]Selecting {card_name} (index {idx})[/cyan]")

            send_failed = False
            if not self.dry_run:
                try:
                    self._execute_with_retry("select_deck_card", option_index=idx)
                except Exception as e:
                    # IMPORTANT: An HTTP timeout here does NOT mean the
                    # action failed on the game side. The mod may have
                    # processed the selection successfully and hung on
                    # an animation / save-game hook. We MUST refresh
                    # game state before deciding what to do next —
                    # otherwise we risk double-selecting, missing a
                    # successful selection, or shifting indices
                    # underneath ourselves. Fall through to the
                    # ground-truth check below instead of `continue`.
                    self._log_action(
                        f"  [yellow]Select send failed: {e} "
                        f"— refreshing state[/yellow]"
                    )
                    send_failed = True

            time.sleep(1.0)  # Wait for the game to process

            try:
                gs = self.client.get_state()
            except Exception:
                return
            self.game_state = gs
            actions = gs.get("available_actions", [])

            # Check if confirm appeared or screen changed — this is the
            # ground-truth "we're done" signal. It fires regardless of
            # whether the previous select_deck_card call got a clean
            # HTTP response.
            if "confirm_selection" in actions:
                if not self.dry_run:
                    try:
                        self._execute_with_retry("confirm_selection")
                    except Exception:
                        pass
                self._log_action("  [dim]auto: confirm_selection[/dim]")
                return
            if "select_deck_card" not in actions:
                return  # Screen changed, done

            # Ground-truth check against the game's own counter. This
            # works whether the previous send succeeded or timed out.
            sel = gs.get("selection") or {}
            curr = sel.get("selected_count", 0)
            if curr > picked:
                picked = curr
                # The selection landed (even if the HTTP call looked
                # like it failed). Clear the send-failed flag so we
                # don't double-log it.
                send_failed = False

            if send_failed:
                # The send raised AND the game's counter didn't
                # advance AND we didn't see confirm/screen-change.
                # Treat this idx as genuinely untaken; the next loop
                # iteration will try the next priority index.
                self._log_action(
                    f"  [dim]index {idx} did not land — "
                    f"trying next priority[/dim]"
                )

        # If we exhausted attempts without the screen changing,
        # mark as stuck so the main loop can time out
        if picked < num_to_pick:
            self._log_action(f"  [yellow]Multi-select stuck (picked {picked}/{num_to_pick})[/yellow]")
            self._deck_select_stuck = True
            self._stuck_since = time.monotonic()

    # ------------------------------------------------------------------
    # Game over
    # ------------------------------------------------------------------

    def _handle_game_over(self) -> None:
        gs = self.game_state
        run = gs.get("run") or {}
        game_over = gs.get("game_over") or {}
        outcome = game_over.get("outcome", "unknown")
        floor = run.get("floor", "?")
        hp = run.get("current_hp", 0)

        from .run_logger import BOSS_FLOORS
        is_boss_floor = isinstance(floor, int) and floor in BOSS_FLOORS

        if outcome == "victory" or hp > 0:
            self._log_action(
                f"[bold green]VICTORY![/bold green] Floor {floor} | HP {hp}"
            )
            self.logger.log_combat_end(gs, "win")
            self.logger.log_run_end(gs, "victory")
            result = "victory"
        elif is_boss_floor:
            self._log_action(
                f"[bold red]BOSS DEFEAT[/bold red] Floor {floor} | HP {hp}"
            )
            self.logger.log_combat_end(gs, "boss_defeat")
            self.logger.log_run_end(gs, "boss_defeat")
            result = "boss_defeat"
        else:
            self._log_action(
                f"[bold red]DEFEAT[/bold red] Floor {floor} | HP {hp}"
            )
            self.logger.log_combat_end(gs, "defeat")
            self.logger.log_run_end(gs, "defeat")
            result = "defeat"

        if self._store_run_started:
            self.store.log_combat_end(
                self._store_run_id,
                floor=floor if isinstance(floor, int) else 0,
                hp=hp, max_hp=run.get("max_hp", 0),
                outcome="win" if result == "victory" else "defeat",
                turns=self.turn_count,
            )
            self.store.end_run(
                self._store_run_id, outcome=result,
                floor=floor if isinstance(floor, int) else 0,
                hp=hp, max_hp=run.get("max_hp"),
            )
            self.store.flush()
            self._store_run_started = False

        self._log_action(
            f"Turns: {self.turn_count} | Actions: {self.action_count}"
        )
        self._refresh()

        # Return to main menu so the next run can start
        actions = gs.get("available_actions", [])
        if "return_to_main_menu" in actions and not self.dry_run:
            time.sleep(1.0)
            try:
                self._execute_with_retry("return_to_main_menu")
                time.sleep(2.0)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Action execution with retry
    # ------------------------------------------------------------------

    def _wait_for_ready(
        self, timeout: float = 15.0, poll: float = 0.25, min_wait: float = 0.3,
    ) -> None:
        """Poll game state until actions are available (player can act).

        The game always responds 200 to GET /state, so we check whether
        available_actions is non-empty to know the game is ready for input.
        min_wait gives animations time to start before we begin polling.
        """
        time.sleep(min_wait)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                gs = self.client.get_state()
                actions = gs.get("available_actions", [])
                screen = gs.get("screen", "")
                # Ready if: player has actions, or we left combat, or game over
                if actions or screen == "GAME_OVER":
                    return
                time.sleep(poll)
            except Exception:
                return
        # Timeout — proceed anyway

    def _execute_with_retry(
        self,
        action: str,
        *,
        card_index: int | None = None,
        target_index: int | None = None,
        option_index: int | None = None,
        retries: int = 10,
        delay: float = 0.3,
    ) -> dict:
        """Execute a game action, retrying on retriable 409 errors.

        The game mod returns 409 for both "action not available in current
        state" (retriable — game is animating) and permanent errors like
        "invalid_target" or "card cannot be played" (not retriable).
        """
        # Truly permanent 409 errors — never retry these.
        # "cannot be played" was previously treated as transient (animation
        # timing), but cards with unmet conditions (e.g. Grand Finale when
        # draw pile isn't empty) will NEVER become playable on the same
        # game state. Retrying just wastes 10 attempts and then the bot
        # dies. Treat it as permanent so the caller can re-plan.
        _PERMANENT_409 = (
            "invalid_target", "out of range",
            "is locked", "out of stock", "not supported",
            "cannot be played",
        )
        # Transient 409 errors — game is animating or action state shifted.
        _TRANSIENT_409 = (
            "not available",
        )
        # HTTP timeouts from the mod — see note below. Cap timeout retries
        # at 2 (not `retries`) because if the mod is hung past a couple of
        # 10s timeouts, we're almost certainly in a worse state than the
        # caller can recover from with more retries. The caller should
        # refresh game state on None/{} return and decide what to do.
        _TIMEOUT_RETRY_LIMIT = 2

        timeout_attempts = 0
        for attempt in range(retries + 1):
            try:
                return self.client.execute_action(
                    action,
                    card_index=card_index,
                    target_index=target_index,
                    option_index=option_index,
                )
            except ConnectionError as e:
                err_str = str(e)
                # HTTP timeouts — the mod didn't respond within its 10s
                # window. IMPORTANT: a timeout does NOT mean the action
                # failed on the game side; the mod may have processed it
                # and hung on a follow-up animation. Callers that care
                # about ground truth should re-check the game state on
                # return. Retry a bounded number of times to handle
                # purely-transient slow responses (save-game hooks,
                # animation stalls), then raise so the caller can take
                # corrective action.
                if "Timed out" in err_str:
                    timeout_attempts += 1
                    if timeout_attempts <= _TIMEOUT_RETRY_LIMIT:
                        wait = min(delay * (1.5 ** (timeout_attempts - 1)), 2.0)
                        self._log_action(
                            f"  [dim]timeout on {action}, retry "
                            f"{timeout_attempts}/{_TIMEOUT_RETRY_LIMIT}[/dim]"
                        )
                        time.sleep(wait)
                        continue
                    # Exhausted timeout retries — raise so callers can
                    # refresh state and decide whether the action
                    # actually landed or not.
                    self._log_action(
                        f"  [yellow]timeout after {_TIMEOUT_RETRY_LIMIT} "
                        f"retries: {err_str[:80]}[/yellow]"
                    )
                    raise
                if "409" not in err_str:
                    raise
                # Permanent errors — don't retry
                if any(kw in err_str for kw in _PERMANENT_409):
                    self._log_action(
                        f"  [yellow]rejected: {err_str[:240]}[/yellow]"
                    )
                    return {}
                # Transient / animation timing — retry with backoff, but
                # re-validate against current game state first. If the
                # action is no longer in available_actions, the game has
                # moved on (animation landed, screen changed, potion gone)
                # and retrying would be a no-op at best and wrong at
                # worst. Bail out early and let the caller re-plan.
                if attempt < retries:
                    try:
                        gs = self.client.get_state()
                        avail = gs.get("available_actions", []) or []
                        if action not in avail:
                            self._log_action(
                                f"  [yellow]{action} no longer available "
                                f"after 409 (screen="
                                f"{gs.get('screen', '?')}); skipping[/yellow]"
                            )
                            self.game_state = gs
                            return {}
                    except Exception:
                        # State fetch failed — fall through to retry
                        pass
                    wait = min(delay * (1.5 ** attempt), 2.0)
                    time.sleep(wait)
                    continue
                # Exhausted retries — log full body (not truncated at 80)
                self._log_action(
                    f"  [yellow]skipped after {retries} retries: "
                    f"{err_str[:240]}[/yellow]"
                )
                return {}


def _load_env_from_mcp_json() -> None:
    """Load env vars from .mcp.json if present."""
    candidate = Path(__file__).resolve().parents[3] / ".mcp.json"
    if not candidate.exists():
        candidate = Path.cwd() / ".mcp.json"
    if not candidate.exists():
        return
    try:
        with open(candidate, encoding="utf-8") as f:
            data = json.load(f)
        for server in data.get("mcpServers", {}).values():
            for key, value in server.get("env", {}).items():
                if key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


def main():
    _load_env_from_mcp_json()
    parser = argparse.ArgumentParser(description="STS2 Autonomous Runner")
    parser.add_argument(
        "--step", action="store_true",
        help="Step mode: press Enter for each action",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show decisions without executing",
    )
    parser.add_argument(
        "--poll", type=float, default=1.0,
        help="Seconds between state polls (default: 1.0)",
    )
    parser.add_argument(
        "--character", type=str, default=DEFAULT_CHARACTER,
        help=f"Character to play (default: {DEFAULT_CHARACTER})",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override advisor model (e.g. qwen3:8b, gemma3:4b)",
    )
    args = parser.parse_args()

    if args.model:
        os.environ["STS2_ADVISOR_MODEL"] = args.model

    runner = Runner(
        step_mode=args.step,
        dry_run=args.dry_run,
        poll_interval=args.poll,
        character=args.character,
    )
    runner.run()


if __name__ == "__main__":
    main()
