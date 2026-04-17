"""Self-play training loop for AlphaZero.

Split into two processes:
    Worker (headless):  python -m sts2_solver.alphazero.self_play train
    Monitor (TUI):      python -m sts2_solver.alphazero.self_play monitor

The worker writes progress to a JSON file that the monitor reads.
The worker runs headless and survives SSH disconnects (use nohup/tmux).
The monitor can be started/stopped freely.

Training loop:
    1. Play N games using MCTS with current network
    2. Collect (state_tensors, mcts_policy, game_outcome) for each turn
    3. Train network on collected data for E epochs
    4. Repeat
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from ..actions import Action, END_TURN, enumerate_actions
from ..combat_engine import (
    end_turn,
    is_combat_over,
    play_card,
    resolve_enemy_intents,
    start_turn,
    tick_enemy_powers,
)
from ..constants import CardType
from ..data_loader import CardDB, load_cards
from ..models import Card, CombatState, EnemyState, PlayerState
from ..simulator import (
    _ensure_data_loaded,
    _ENCOUNTERS_BY_ID,
    _spawn_enemy,
    _create_enemy_ai,
    _set_enemy_intents,
    _resolve_sim_intents,
    ENEMY_MOVE_TABLES,
)

from .encoding import EncoderConfig, Vocabs, build_vocabs_from_card_db
from .mcts import MCTS, scale_simulations
from .network import STS2Network
from .state_tensor import encode_state, encode_actions


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

@dataclass
class TrainingSample:
    """One training sample from a self-play game."""
    state_tensors: dict[str, torch.Tensor]
    policy: list[float]
    value: float
    action_card_ids: torch.Tensor
    action_features: torch.Tensor
    action_mask: torch.Tensor
    num_actions: int
    wasted_energy: bool = False  # True if this was an END_TURN with playable cards
    value_penalty: float = 0.0   # Additional penalty applied when value is assigned


@dataclass
class OptionSample:
    """Training sample for non-combat decisions (rest/map/shop)."""
    state_tensors: dict[str, torch.Tensor]
    option_types: list[int]   # Option type indices (see OPTION_* constants)
    option_cards: list[int]   # Card vocab indices (0 when N/A)
    chosen_idx: int           # Which option was picked
    value: float              # Run outcome value (assigned after run ends)
    # Shadow pick from the deterministic advisor computed at the same
    # decision site. Used in training loss (Approach 2: shadow advisor
    # signal) to push the shadow's preferred option toward the run value,
    # plus consumed by tools/agreement_rate.py for diagnostics.
    shadow_chosen_idx: int | None = None
    # Deck card vocab IDs at decision time — used by the dedicated
    # card_eval_head to encode deck composition context.  None for
    # non-card-pick decisions (rest, map, shop, events).
    deck_card_ids: list[int] | None = None


# Option type constants (indices into option_type_embed)
OPTION_REST = 1
OPTION_SMITH = 2
OPTION_MAP_WEAK = 3
OPTION_MAP_NORMAL = 4
OPTION_MAP_ELITE = 5
OPTION_MAP_EVENT = 6
OPTION_MAP_SHOP = 7
OPTION_MAP_REST = 8
OPTION_SHOP_REMOVE = 9
OPTION_SHOP_BUY = 10
OPTION_SHOP_LEAVE = 11
OPTION_CARD_REWARD = 12
OPTION_CARD_SKIP = 13
OPTION_SHOP_BUY_POTION = 14
# IMPROVEMENTS.md #4: event-choice decisions get their own option type so
# the option-head network can learn per-event policies from outcome value.
# opt_cards[i] carries the per-choice vocab id (see EVENT_CHOICE_VOCAB in
# simulator.py). A new event seen for the first time maps to 0 (UNK) and
# still gets a reasonable prior from the side-features wired through.
OPTION_EVENT_CHOICE = 15
# IMPROVEMENTS.md #18: shop relics were invisible to the option head.
# Now enumerated in full_run's shop loop and bridge.shop_options_from_mcp.
# opt_cards[i] carries the relic's vocab index from vocabs.relics.
OPTION_SHOP_BUY_RELIC = 16

ROOM_TYPE_TO_OPTION = {
    "weak": OPTION_MAP_WEAK,
    "normal": OPTION_MAP_NORMAL,
    "elite": OPTION_MAP_ELITE,
    "event": OPTION_MAP_EVENT,
    "shop": OPTION_MAP_SHOP,
    "rest": OPTION_MAP_REST,
}


_JUNK_TYPES = {CardType.STATUS, CardType.CURSE}


def _affordable_play_actions(
    actions: list[Action], state: CombatState,
) -> list[Action]:
    """Return playable non-junk card actions the player can afford right now."""
    hand = state.player.hand
    energy = state.player.energy
    affordable = []
    for a in actions:
        if a.action_type != "play_card" or a.card_idx is None:
            continue
        if a.card_idx >= len(hand):
            continue
        card = hand[a.card_idx]
        if card.card_type in _JUNK_TYPES:
            continue
        if card.cost <= energy:
            affordable.append(a)
    return affordable


class ReplayBuffer:
    """Fixed-size buffer with separate win reservoir for prioritized replay.

    Maintains a main FIFO buffer plus a dedicated win buffer that preserves
    samples from winning games.  When sampling, ``win_mix_ratio`` of the
    batch is drawn from the win buffer (if available) so the network always
    sees positive signal even when wins are rare.
    """

    def __init__(self, capacity: int = 50_000, win_capacity: int = 10_000,
                 win_mix_ratio: float = 0.25):
        self.buffer: deque[TrainingSample] = deque(maxlen=capacity)
        self.win_buffer: deque[TrainingSample] = deque(maxlen=win_capacity)
        self.win_mix_ratio = win_mix_ratio

    def add(self, sample: TrainingSample, is_win: bool = False) -> None:
        self.buffer.append(sample)
        if is_win:
            self.win_buffer.append(sample)

    def sample(self, batch_size: int) -> list[TrainingSample]:
        if len(self.win_buffer) > 0 and self.win_mix_ratio > 0:
            n_win = max(1, int(batch_size * self.win_mix_ratio))
            n_main = batch_size - n_win
            win_samples = random.sample(
                list(self.win_buffer), min(n_win, len(self.win_buffer)))
            main_samples = random.sample(
                list(self.buffer), min(n_main, len(self.buffer)))
            return win_samples + main_samples
        return random.sample(list(self.buffer), min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Self-play game
# ---------------------------------------------------------------------------

# All Act 1 encounters for training. IDs must match encounters.json exactly.
#
# PREVIOUSLY: Used ENCOUNTER_* IDs that didn't exist in the game data,
# causing the training to fall back to the first 5 alphabetical encounters
# (Act 3 Axebots, event encounter, Act 2 Bowlbugs, etc.) — completely
# misaligned with what the bot actually faces in Act 1.
#
# NOW: Every Act 1 encounter by real game data ID, grouped by difficulty.
# The self-play loop picks randomly from this list each game.
TRAINING_ENCOUNTERS = [
    # ── Weak encounters (floors 1-3, easy early fights) ──
    "NIBBITS_WEAK",                    # Single Nibbit
    "SHRINKER_BEETLE_WEAK",            # Single Shrinker Beetle
    "FUZZY_WURM_CRAWLER_WEAK",         # Single Fuzzy Wurm Crawler
    "SLIMES_WEAK",                     # Leaf Slime (M/S) + Twig Slime (M/S)
    "CORPSE_SLUGS_WEAK",              # Single Corpse Slug
    "EXOSKELETONS_WEAK",              # Single Exoskeleton
    "SCROLLS_OF_BITING_WEAK",         # Single Scroll of Biting
    "SEAPUNK_WEAK",                    # Single Seapunk
    "SLUDGE_SPINNER_WEAK",            # Single Sludge Spinner
    "TUNNELER_WEAK",                   # Single Tunneler
    "TOADPOLES_WEAK",                  # Toadpoles
    "THIEVING_HOPPER_WEAK",           # Single Thieving Hopper
    "DEVOTED_SCULPTOR_WEAK",          # Single Devoted Sculptor
    "BOWLBUGS_WEAK",                   # 3 Bowlbugs

    # ── Normal encounters (floors 4-12, the bulk of Act 1) ──
    "NIBBITS_NORMAL",                  # Nibbit (stronger variant)
    "SLIMES_NORMAL",                   # 4 slimes (stronger variant)
    "RUBY_RAIDERS_NORMAL",             # 5 raiders — multi-enemy, intent-varied
    "INKLETS_NORMAL",                  # Multiple Inklets
    "MAWLER_NORMAL",                   # Single tanky enemy
    "CUBEX_CONSTRUCT_NORMAL",          # Single construct
    "VINE_SHAMBLER_NORMAL",            # Single shambler
    "FLYCONID_NORMAL",                 # Flyconid + 2 slimes (mixed group)
    "SNAPPING_JAXFRUIT_NORMAL",        # Jaxfruit + Flyconid
    "FOGMOG_NORMAL",                   # Eye With Teeth + Fogmog
    "OVERGROWTH_CRAWLERS",             # Fuzzy Wurm Crawler + Shrinker Beetle
    "SLITHERING_STRANGLER_NORMAL",     # 6-enemy fight — hardest normal encounter
    "CHOMPERS_NORMAL",                 # Chomper — blocks + debuffs
    "BOWLBUGS_NORMAL",                 # 4 Bowlbugs — mixed group
    "CORPSE_SLUGS_NORMAL",            # Corpse Slugs
    "CONSTRUCT_MENAGERIE_NORMAL",      # Cubex + Punch Construct
    "CULTISTS_NORMAL",                 # Calcified + Damp Cultist
    "FOSSIL_STALKER_NORMAL",           # Fossil Stalker
    "FROG_KNIGHT_NORMAL",              # Frog Knight — high damage
    "LOUSE_PROGENITOR_NORMAL",         # Louse Progenitor — tanky
    "LIVING_FOG_NORMAL",               # Living Fog + Gas Bomb
    "TWO_TAILED_RATS_NORMAL",         # Two Tailed Rats
    "PUNCH_CONSTRUCT_NORMAL",          # Punch Construct — charge up + big hits
    "SPINY_TOAD_NORMAL",              # Spiny Toad — thorns + AoE
    "HUNTER_KILLER_NORMAL",            # Hunter Killer — debuff + multi-hit
    "OWL_MAGISTRATE_NORMAL",           # Owl Magistrate — debuffs + tanky
    "SLIMED_BERSERKER_NORMAL",         # Slimed Berserker — ramps strength
    "MYTES_NORMAL",                    # Mytes — small swarm
    "AXEBOTS_NORMAL",                  # Axebots
    "HAUNTED_SHIP_NORMAL",             # Haunted Ship — multi-hit + debuffs
    "SEWER_CLAM_NORMAL",              # Sewer Clam — blocks + attacks
    "THE_LOST_AND_FORGOTTEN_NORMAL",   # The Lost + The Forgotten
    "THE_OBSCURA_NORMAL",             # The Obscura — heavy debuffs
    "OVICOPTER_NORMAL",                # Ovicopter + Tough Egg
    "EXOSKELETONS_NORMAL",             # Exoskeletons (normal variant)
    "SCROLLS_OF_BITING_NORMAL",        # Scrolls of Biting (normal)
    "TOADPOLES_NORMAL",                # Toadpoles + Calcified Cultist
    "FABRICATOR_NORMAL",               # Fabricator — ramps + big hit
    "GLOBE_HEAD_NORMAL",               # Globe Head

    # ── Elites (high-threat fights) ──
    "BYRDONIS_ELITE",                  # Single elite — high damage
    "BYGONE_EFFIGY_ELITE",             # Single elite — 0% win rate in logs
    "PHROG_PARASITE_ELITE",            # Phrog Parasite + Wriggler
    "DECIMILLIPEDE_ELITE",             # 3-segment segmented enemy
    "ENTOMANCER_ELITE",                # Summons + multi-hit + big finisher
    "SKULKING_COLONY_ELITE",           # Swarm + crush
    "MECHA_KNIGHT_ELITE",              # Shield bash + triple strike + overclock
    "INFESTED_PRISMS_ELITE",           # Beam + haze + crystal shell
    "TERROR_EEL_ELITE",                # Electric multi-hit + terrify
    "SOUL_NEXUS_ELITE",                # Soul drain + spirit barrage
    "PHANTASMAL_GARDENERS_ELITE",      # Vine lash + thorn storm
    "KNIGHTS_ELITE",                   # 3-knight team (Flail + Magi + Spectral)

    # ── Bosses ──
    "VANTOM_BOSS",                     # Act 1 boss — long fight
    "CEREMONIAL_BEAST_BOSS",           # Act 1 boss — high HP
    "THE_KIN_BOSS",                    # Act 1 boss — Kin Follower + Kin Priest
    "DOORMAKER_BOSS",                  # Doormaker + Door
    "WATERFALL_GIANT_BOSS",            # High HP, cascade + tidal crush
    "LAGAVULIN_MATRIARCH_BOSS",        # Sleeps then bursts, strength ramp
    "KNOWLEDGE_DEMON_BOSS",            # Heavy debuffs + mind rend
    "KAISER_CRAB_BOSS",                # Crusher + Rocket duo
    "QUEEN_BOSS",                      # Queen + Torch Head Amalgam
    "SOUL_FYSH_BOSS",                  # Soul siphon + bubble barrage
    "TEST_SUBJECT_BOSS",               # Mutates + frenzy + annihilate
    "THE_INSATIABLE_BOSS",             # Devour + feast — highest damage boss
]


def _make_starter_deck(card_db: CardDB, character: str = "silent") -> list[Card]:
    """Build a basic starter deck."""
    cards = []
    strike = card_db.get("STRIKE_SILENT") or card_db.get("STRIKE")
    defend = card_db.get("DEFEND_SILENT") or card_db.get("DEFEND")
    neutralize = card_db.get("NEUTRALIZE")
    survivor = card_db.get("SURVIVOR")

    if strike:
        cards.extend([strike] * 5)
    if defend:
        cards.extend([defend] * 5)
    if neutralize:
        cards.append(neutralize)
    if survivor:
        cards.append(survivor)
    return cards


def play_one_game(
    mcts: MCTS,
    card_db: CardDB,
    vocabs: Vocabs,
    config: EncoderConfig,
    encounter_id: str | None = None,
    deck: list[Card] | None = None,
    max_turns: int = 30,
    mcts_simulations: int = 100,
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> tuple[list[TrainingSample], str, int, str]:
    """Play one combat game using MCTS.

    Returns (samples, outcome, turns, encounter_id).
    """
    if rng is None:
        rng = random.Random()

    _ensure_data_loaded()

    if encounter_id is None:
        available = [e for e in TRAINING_ENCOUNTERS if e in _ENCOUNTERS_BY_ID]
        if not available:
            available = list(_ENCOUNTERS_BY_ID.keys())[:5]
        encounter_id = rng.choice(available)

    if deck is None:
        deck = _make_starter_deck(card_db)

    enc = _ENCOUNTERS_BY_ID.get(encounter_id, {})
    monster_list = enc.get("monsters", [])
    enemies: list[EnemyState] = []
    enemy_ais = []
    for m in monster_list:
        mid = m["id"]
        enemy = _spawn_enemy(mid)
        enemies.append(enemy)
        enemy_ais.append(_create_enemy_ai(mid))

    if not enemies:
        return [], "win", 0, encounter_id

    draw_pile = list(deck)
    rng.shuffle(draw_pile)
    player = PlayerState(
        hp=70, max_hp=70, energy=3, max_energy=3,
        draw_pile=draw_pile,
    )
    state = CombatState(player=player, enemies=enemies)
    samples: list[TrainingSample] = []
    turn_count = 0
    outcome = None

    player_max_hp = state.player.max_hp

    for turn_num in range(1, max_turns + 1):
        start_turn(state)
        turn_count = turn_num
        _set_enemy_intents(state, enemy_ais)

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
            action_features, action_mask = encode_actions(actions, state, vocabs, config)

            scaled_sims = scale_simulations(mcts_simulations, len(actions))
            action, policy, _root_value = mcts.search(
                state, num_simulations=scaled_sims,
                temperature=temperature,
            )

            # -- Force-play override: if MCTS wants END_TURN but there are
            # affordable non-junk cards in hand, override with a random
            # playable card 80% of the time during training.  This generates
            # exploration data so the network learns the value of playing
            # affordable cards vs ending early.
            wasted = False
            if action.action_type == "end_turn":
                affordable = _affordable_play_actions(actions, state)
                if affordable:
                    wasted = True
                    if rng.random() < 0.80:
                        action = rng.choice(affordable)

            samples.append(TrainingSample(
                state_tensors=state_tensors,
                policy=policy,
                value=0.0,
                action_features=action_features,
                action_mask=action_mask,
                num_actions=len(actions),
                wasted_energy=wasted,
            ))

            if action.action_type == "end_turn":
                break

            if action.action_type == "choose_card":
                # Resolve pending choice — doesn't count as a card play
                if action.choice_idx is not None and state.pending_choice is not None:
                    from ..effects import discard_card_from_hand
                    pc = state.pending_choice
                    if pc.choice_type == "discard_from_hand":
                        if action.choice_idx < len(state.player.hand):
                            discard_card_from_hand(state, action.choice_idx)
                        pc.chosen_so_far.append(action.choice_idx)
                        if len(pc.chosen_so_far) >= pc.num_choices:
                            state.pending_choice = None
            elif action.card_idx is not None:
                from ..combat_engine import can_play_card
                if can_play_card(state, action.card_idx):
                    play_card(state, action.card_idx, action.target_idx, card_db)
                    cards_this_turn += 1

            outcome = is_combat_over(state)
            if outcome:
                break

        turn_end_sample = len(samples)

        if outcome:
            break

        # -- Intent-aware reward signals --
        incoming_damage = 0
        for e in state.enemies:
            if e.is_alive and e.intent_type == "Attack" and e.intent_damage:
                incoming_damage += e.intent_damage * max(1, e.intent_hits)

        hp_before_enemy = state.player.hp
        player_block_played = state.player.block
        hand_block_available = sum(
            c.block for c in state.player.hand
            if c.block and c.block > 0 and c.cost <= state.player.energy
        )
        hand_damage_available = sum(
            c.damage for c in state.player.hand
            if c.damage and c.damage > 0 and c.cost <= state.player.energy
        )

        # Signal 1: Offensive-when-safe
        if (incoming_damage == 0 and player_block_played > 0
                and turn_start_sample < turn_end_sample):
            safe_block_penalty = min(0.08, player_block_played / max(1, player_max_hp) * 0.3)
            for idx in range(turn_start_sample, turn_end_sample):
                samples[idx].value_penalty += safe_block_penalty

        # Signal 2: Lethal-awareness
        for e in state.enemies:
            if e.is_alive and e.hp > 0 and e.hp <= hand_damage_available:
                lethal_penalty = min(0.10, e.hp / max(1, player_max_hp) * 0.25)
                for idx in range(turn_start_sample, turn_end_sample):
                    samples[idx].value_penalty += lethal_penalty
                break

        end_turn(state)
        resolve_enemy_intents(state)
        _resolve_sim_intents(state, enemy_ais)
        tick_enemy_powers(state)

        hp_after_enemy = state.player.hp
        damage_taken = max(0, hp_before_enemy - hp_after_enemy)

        # Signal 3: Intent-weighted blocking
        if turn_start_sample < turn_end_sample and incoming_damage > 0:
            if player_block_played < incoming_damage:
                gap = incoming_damage - player_block_played
                under_penalty = min(0.15, gap / max(1, player_max_hp) * 0.4)
                if hand_block_available > 0:
                    for idx in range(turn_start_sample, turn_end_sample):
                        samples[idx].value_penalty += under_penalty
            elif player_block_played > incoming_damage * 1.5:
                excess = player_block_played - incoming_damage
                over_penalty = min(0.06, excess / max(1, player_max_hp) * 0.15)
                for idx in range(turn_start_sample, turn_end_sample):
                    samples[idx].value_penalty += over_penalty

        outcome = is_combat_over(state)
        if outcome:
            break

    if outcome is None:
        outcome = "lose"

    # Value based on HP remaining, not binary win/loss.
    # Win with full HP = +1.0, win with 1 HP = ~+0.5
    # Lose = scaled by how much HP was remaining (-0.2 to -1.0)
    # This gives much richer training signal than binary +1/-1.
    hp_frac = state.player.hp / max(1, state.player.max_hp)
    if outcome == "win":
        value = 0.5 + 0.5 * hp_frac  # [0.5, 1.0]
    else:
        value = -0.5 - 0.5 * (1.0 - hp_frac)  # [-1.0, -0.5]

    for sample in samples:
        sample.value = value
        # Per-step penalties: wasted energy + any block penalty
        penalty = sample.value_penalty
        if sample.wasted_energy:
            penalty += 0.15
        if penalty > 0:
            sample.value = max(-1.0, sample.value - penalty)

    return samples, outcome, turn_count, encounter_id


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_batch(
    network: STS2Network,
    optimizer: torch.optim.Optimizer,
    samples: list[TrainingSample],
    option_samples: list | None = None,
    device: str = "cpu",
) -> tuple[float, float, float, float]:
    """Train on a batch. Returns (total, value, policy, option) losses."""
    network.train()
    value_losses = []
    policy_losses = []
    option_losses = []
    nan_combat = nan_option = 0

    # --- Combat samples: accumulate gradients, step once ---
    optimizer.zero_grad()
    valid_count = 0

    for sample in samples:
        state_tensors = {k: v.to(device) for k, v in sample.state_tensors.items()}
        action_card_ids = sample.action_card_ids.to(device)
        action_features = sample.action_features.to(device)
        action_mask = sample.action_mask.to(device)

        hidden = network.encode_state(**state_tensors)
        value, logits = network.forward(hidden, action_card_ids, action_features, action_mask)

        target_value = torch.tensor([[sample.value]], dtype=torch.float32, device=device)
        v_loss = F.mse_loss(value, target_value)

        target_policy = torch.tensor(
            sample.policy[:sample.num_actions], dtype=torch.float32, device=device
        )
        if len(target_policy) < logits.shape[1]:
            padding = torch.zeros(logits.shape[1] - len(target_policy), device=device)
            target_policy = torch.cat([target_policy, padding])
        log_probs = F.log_softmax(logits[0, :len(sample.policy)], dim=0)
        p_loss = -torch.sum(target_policy[:len(log_probs)] * log_probs)

        loss = 0.5 * v_loss + p_loss
        if torch.isnan(loss):
            nan_combat += 1
            continue
        value_losses.append(v_loss.item())
        policy_losses.append(p_loss.item())
        # Accumulate gradients without growing the graph (#8)
        (loss / max(1, len(samples))).backward()
        valid_count += 1

    if valid_count > 0:
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()

    # --- Option samples (all non-combat decisions): accumulate gradients, step once ---
    # Card-pick samples (deck_card_ids != None) are routed through the
    # dedicated card_eval_head with ranking loss.  All other option samples
    # (rest, map, shop, events) use the generic option_eval_head as before.
    optimizer.zero_grad()
    option_valid = 0
    SHADOW_ALPHA = 0.15
    RANK_BETA = 0.10       # weight for ranking loss on card picks
    RANK_MARGIN = 0.05     # minimum desired score gap between chosen and alternatives
    for sample in (option_samples or []):
        try:
            state_tensors = {k: v.to(device) for k, v in sample.state_tensors.items()}
            hidden = network.encode_state(**state_tensors)

            max_card_id = network.card_embed.num_embeddings - 1
            clamped_cards = [c if c <= max_card_id else 1 for c in sample.option_cards]  # 1=UNK
            types_t = torch.tensor([sample.option_types], dtype=torch.long, device=device)
            cards_t = torch.tensor([clamped_cards], dtype=torch.long, device=device)
            opt_mask = torch.zeros(1, len(sample.option_types), dtype=torch.bool, device=device)

            target = torch.tensor([[sample.value]], dtype=torch.float32, device=device)

            # ---- Card-pick samples: dedicated head + ranking loss ----
            if sample.deck_card_ids is not None:
                deck_ids = [min(d, max_card_id) if d is not None else 1 for d in sample.deck_card_ids]
                deck_t = torch.tensor([deck_ids], dtype=torch.long, device=device)
                deck_mask = torch.zeros(1, len(deck_ids), dtype=torch.bool, device=device)

                scores = network.evaluate_card_picks(
                    hidden, deck_t, deck_mask, types_t, cards_t, opt_mask)

                # Primary loss: MSE on chosen option's score → run value
                chosen_score = scores[0, sample.chosen_idx].unsqueeze(0).unsqueeze(0)
                o_loss = 0.25 * F.mse_loss(chosen_score, target)

                # Ranking loss: chosen should beat alternatives (good run)
                # or alternatives should beat chosen (bad run).
                num_opts = scores.shape[1]
                if num_opts > 1:
                    rank_loss = torch.tensor(0.0, device=device)
                    chosen_s = scores[0, sample.chosen_idx]
                    for j in range(num_opts):
                        if j == sample.chosen_idx:
                            continue
                        other_s = scores[0, j]
                        if sample.value > 0.5:
                            # Good run: chosen should score > alternative by margin
                            rank_loss = rank_loss + F.relu(RANK_MARGIN - (chosen_s - other_s))
                        else:
                            # Bad run: alternative should score > chosen by margin
                            rank_loss = rank_loss + F.relu(RANK_MARGIN - (other_s - chosen_s))
                    rank_loss = rank_loss / (num_opts - 1)
                    o_loss = o_loss + RANK_BETA * rank_loss

            # ---- All other options: generic option_eval_head ----
            else:
                scores = network.evaluate_options(hidden, types_t, cards_t, opt_mask)

                chosen_score = scores[0, sample.chosen_idx].unsqueeze(0).unsqueeze(0)
                o_loss = 0.25 * F.mse_loss(chosen_score, target)

            # ----------------------------------------------------------
            # Approach 2: Shadow advisor signal (all option types)
            # When the shadow (heuristic) advisor disagrees with the
            # network's choice, also push the shadow's preferred option
            # toward the run outcome value.
            # ----------------------------------------------------------
            if (sample.shadow_chosen_idx is not None
                    and sample.shadow_chosen_idx != sample.chosen_idx
                    and sample.shadow_chosen_idx < scores.shape[1]):
                shadow_score = scores[0, sample.shadow_chosen_idx].unsqueeze(0).unsqueeze(0)
                shadow_loss = SHADOW_ALPHA * F.mse_loss(shadow_score, target)
                o_loss = o_loss + shadow_loss

            if torch.isnan(o_loss):
                nan_option += 1
                continue
            option_losses.append(o_loss.item())
            n_opt = max(1, len(option_samples or []))
            (o_loss / n_opt).backward()
            option_valid += 1
        except Exception:
            continue

    if option_valid > 0:
        torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()

    total_nan = nan_combat + nan_option
    if total_nan > 0:
        print(f"  [warn] NaN losses skipped: combat={nan_combat} option={nan_option}", flush=True)

    avg_v = sum(value_losses) / max(1, len(value_losses))
    avg_p = sum(policy_losses) / max(1, len(policy_losses))
    avg_o = sum(option_losses) / max(1, len(option_losses))
    return avg_v + avg_p + avg_o, avg_v, avg_p, avg_o


# ---------------------------------------------------------------------------
# Progress file (shared between worker and monitor)
# ---------------------------------------------------------------------------

def _default_progress_path() -> Path:
    return Path(__file__).resolve().parents[4] / "alphazero_progress.json"


def _write_progress(path: Path, stats: dict) -> None:
    """Atomically write progress to JSON file."""
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    tmp.replace(path)


def _read_progress(path: Path) -> dict:
    """Read progress from JSON file."""
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------------
# Worker: headless training loop
# ---------------------------------------------------------------------------

def train_worker(
    num_generations: int = 100,
    games_per_generation: int = 7,
    mcts_simulations: int = 100,
    batch_size: int = 64,
    train_epochs: int = 3,
    lr: float = 1e-3,
    temperature: float = 1.0,
    save_dir: str | None = None,
    progress_file: str | None = None,
    boss_log_file: str | None = None,
):
    """Headless training loop. Writes progress to JSON file."""
    card_db = load_cards()
    vocabs = build_vocabs_from_card_db(card_db)
    config = EncoderConfig()
    network = STS2Network(vocabs, config)
    # Exclude embedding tables from weight decay so rare cards/powers
    # can develop strong representations (#12)
    embed_params = [p for n, p in network.named_parameters() if "embed" in n]
    other_params = [p for n, p in network.named_parameters() if "embed" not in n]
    optimizer = Adam([
        {"params": embed_params, "weight_decay": 0},
        {"params": other_params, "weight_decay": 1e-4},
    ], lr=lr)
    # Warm restarts: LR resets every T_0 generations, giving the network
    # periodic chances to escape local minima as its own play improves.
    # T_0=120 means ~9 restart cycles over 1080 gens, each starting at
    # full LR and decaying to eta_min before resetting.
    restart_period = max(60, num_generations // 9)
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=restart_period, T_mult=1, eta_min=5e-5
    )
    replay_buffer = ReplayBuffer(capacity=50_000)
    option_buffer = ReplayBuffer(capacity=15_000)  # All non-combat decisions (cards, rest, map, shop)
    mcts = MCTS(network, vocabs, config, card_db=card_db, device="cpu")

    save_path = Path(save_dir) if save_dir else Path(__file__).resolve().parents[4] / "alphazero_checkpoints"
    save_path.mkdir(parents=True, exist_ok=True)

    # Load latest checkpoint if available (warm start)
    # Filter out keys with shape mismatches (e.g. trunk input dim changed)
    import torch as _torch
    ckpts = sorted(save_path.glob("gen_*.pt"), key=lambda p: p.stat().st_mtime)
    if ckpts:
        ckpt = _torch.load(ckpts[-1], map_location="cpu", weights_only=True)
        saved_state = ckpt["model_state"]
        current_state = network.state_dict()
        compatible = {
            k: v for k, v in saved_state.items()
            if k in current_state and v.shape == current_state[k].shape
        }
        skipped = set(saved_state.keys()) - set(compatible.keys())
        # If trunk.0 was skipped (input dim changed), also skip trunk.2
        # to avoid NaN from mismatched weight expectations
        if any("trunk.0" in k for k in skipped):
            trunk_keys = [k for k in compatible if k.startswith("trunk.")]
            for k in trunk_keys:
                compatible.pop(k)
                skipped.add(k)
        network.load_state_dict(compatible, strict=False)
        msg = f"Warm start from {ckpts[-1].name} ({len(compatible)}/{len(saved_state)} params)"
        if skipped:
            msg += f", skipped {len(skipped)} shape-mismatched"
        print(msg, flush=True)

    progress_path = Path(progress_file) if progress_file else _default_progress_path()

    rng = random.Random(42)
    t_start = time.time()
    total_wins = 0
    total_games = 0
    total_boss_reached = 0   # runs where floor_reached >= BOSS_FLOOR
    total_boss_wins = 0      # runs that beat the boss outright
    BOSS_FLOOR = 17          # Act 1 boss floor
    recent_games: list[dict] = []

    # V8: relic telemetry (cumulative pickups across all runs)
    from collections import Counter as _Counter
    relic_counts: _Counter = _Counter()
    total_relics_seen: int = 0  # total pickups (sum of counter)
    try:
        from .. import relic_effects as _relic_effects
        relic_pool_size = len(_relic_effects.simulated_relic_ids())
    except Exception:
        relic_pool_size = 0

    from .full_run import play_full_run

    # --- Boss-fight detail log (appended JSONL) ---
    # Each line: one run that reached the boss, with per-turn detail.
    # Lets us analyse play patterns, card usage, and loss modes.
    # By default the log sits next to the checkpoints so each training
    # version gets its own log; --boss-log-file overrides.
    if boss_log_file:
        boss_log_path = Path(boss_log_file)
    else:
        boss_log_path = save_path / "boss_fights.jsonl"
    boss_log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Boss-fight log: {boss_log_path}", flush=True)

    sim_early = int(mcts_simulations * 0.4)
    sim_late = int(mcts_simulations * 1.8)
    print(f"AlphaZero training (full runs): {num_generations} generations, {games_per_generation} runs/gen, {mcts_simulations} base sims ({sim_early}→{sim_late} progressive)", flush=True)
    print(f"Checkpoints: {save_path}", flush=True)
    print(f"Progress: {progress_path}", flush=True)

    for gen in range(1, num_generations + 1):
        gen_t0 = time.time()

        # --- Self-play: full Act 1 runs ---
        gen_wins = 0
        progress = gen / num_generations
        for game_num in range(games_per_generation):
            # Temperature: cosine decay, exploration → exploitation.
            # Stays above 0.5 for ~60% of training, floors at 0.2 late.
            game_temp = 0.2 + 0.8 * temperature * (1 + math.cos(math.pi * progress)) / 2

            # Progressive sim scaling: ramp up sims as training progresses.
            # Early gens (network is random): fewer sims are fine (saves compute).
            # Late gens (network is trained): deeper search finds better plays
            # and produces higher-quality policy/value targets.
            # Scales from 40% → 180% of base sims over training.
            # With base=100: early=40 sims, mid=110 sims, late=180 sims.
            sim_scale = 0.4 + 1.4 * progress
            gen_sims = int(mcts_simulations * sim_scale)

            result = play_full_run(
                mcts, card_db, vocabs, config,
                character="SILENT",
                mcts_simulations=gen_sims,
                temperature=game_temp,
                rng=rng,
            )

            is_win = result.outcome == "win"
            for sample in result.samples:
                replay_buffer.add(sample, is_win=is_win)
            for os in result.deck_samples:
                option_buffer.add(os, is_win=is_win)
            for os in result.option_samples:
                option_buffer.add(os, is_win=is_win)

            total_games += 1
            if result.outcome == "win":
                gen_wins += 1
                total_wins += 1

            # Boss-fight tracking: a run that reached floor >= BOSS_FLOOR is
            # a "boss fight attempted"; a run that won is a "boss fight won".
            # (You can only win an Act 1 run by beating the boss, so total
            # wins == total boss wins in practice.)
            if result.floor_reached >= BOSS_FLOOR:
                total_boss_reached += 1
                if result.outcome == "win":
                    total_boss_wins += 1

            # Persist boss-fight detail if the run reached the boss.
            # Written as one JSON object per line (JSONL) so we can stream-parse.
            _boss = getattr(result, "boss_detail", None)
            if _boss is not None:
                try:
                    _entry = {
                        "gen": gen,
                        "game_num": total_games,
                        "run_outcome": result.outcome,
                        "floor_reached": result.floor_reached,
                        "final_deck": getattr(result, "final_deck", None),
                        "archetype": getattr(result, "archetype", "unknown"),
                        "archetype_commitment": round(
                            getattr(result, "archetype_commitment", 0.0), 3),
                        "boss": _boss,
                    }
                    with open(boss_log_path, "a", encoding="utf-8") as _bl:
                        _bl.write(json.dumps(_entry, default=str) + "\n")
                except Exception as _e:
                    # Never crash training because of a log write.
                    print(f"[boss-log] write failed: {_e}", flush=True)

            # V8: record relic pickups for this run (excluding the starter)
            _run_relics = getattr(result, "final_relics", None) or []
            for _rid in _run_relics:
                if _rid == "RING_OF_THE_SNAKE":
                    continue  # starter relic, not informative
                relic_counts[_rid] += 1
                total_relics_seen += 1

            recent_games.append({
                "num": total_games,
                "encounter": f"Act1 ({result.combats_won}/{result.combats_fought})",
                "outcome": result.outcome,
                "floor": result.floor_reached,
                "hp": result.final_hp,
                "archetype": getattr(result, 'archetype', 'unknown'),
                "commitment": round(getattr(result, 'archetype_commitment', 0.0), 2),
                "relics": [r for r in _run_relics if r != "RING_OF_THE_SNAKE"],
            })
            if len(recent_games) > 50:
                recent_games = recent_games[-50:]

        # --- Training ---
        v_loss = p_loss = o_loss = total_loss = 0.0
        if len(replay_buffer) >= batch_size:
            for epoch in range(train_epochs):
                batch = replay_buffer.sample(batch_size)
                option_batch = option_buffer.sample(min(48, len(option_buffer))) if len(option_buffer) > 0 else []
                total_loss, v_loss, p_loss, o_loss = train_batch(
                    network, optimizer, batch,
                    option_samples=option_batch,
                    device="cpu",
                )
            scheduler.step()

        gen_elapsed = time.time() - gen_t0
        total_elapsed = time.time() - t_start
        mins, secs = divmod(int(total_elapsed), 60)
        hours, mins = divmod(mins, 60)

        # Archetype stats from recent games
        _recent_50 = recent_games[-50:]
        _arch_counts = {}
        _arch_wins = {}
        for _g in _recent_50:
            _a = _g.get("archetype", "unknown")
            _arch_counts[_a] = _arch_counts.get(_a, 0) + 1
            if _g["outcome"] == "win":
                _arch_wins[_a] = _arch_wins.get(_a, 0) + 1
        _arch_stats = {}
        for _a, _cnt in _arch_counts.items():
            _wins = _arch_wins.get(_a, 0)
            _arch_stats[_a] = {
                "count": _cnt,
                "wins": _wins,
                "win_rate": round(_wins / max(1, _cnt), 3),
            }

        # --- Boss-fight metrics (cumulative + recent-50 window) ---
        boss_fight_wr = total_boss_wins / max(1, total_boss_reached)
        _recent_boss_reached = sum(
            1 for _g in _recent_50 if _g.get("floor", 0) >= BOSS_FLOOR
        )
        _recent_boss_wins = sum(
            1 for _g in _recent_50
            if _g.get("floor", 0) >= BOSS_FLOOR and _g.get("outcome") == "win"
        )
        recent_boss_fight_wr = _recent_boss_wins / max(1, _recent_boss_reached)

        # Write progress
        stats = {
            "generation": gen,
            "num_generations": num_generations,
            "games_played": total_games,
            "win_rate": total_wins / max(1, total_games),
            "gen_win_rate": gen_wins / max(1, games_per_generation),
            "boss_fights_reached": total_boss_reached,
            "boss_fights_won": total_boss_wins,
            "boss_fight_win_rate": round(boss_fight_wr, 4),
            "recent_boss_fights_reached": _recent_boss_reached,
            "recent_boss_fights_won": _recent_boss_wins,
            "recent_boss_fight_win_rate": round(recent_boss_fight_wr, 4),
            "buffer_size": len(replay_buffer),
            "total_loss": round(total_loss, 4),
            "value_loss": round(v_loss, 4),
            "policy_loss": round(p_loss, 4),
            "option_loss": round(o_loss, 4),
            "option_buffer_size": len(option_buffer),
            "lr": round(scheduler.get_last_lr()[0], 6),
            "mcts_sims": mcts_simulations,
            "games_per_gen": games_per_generation,
            "elapsed": f"{hours}:{mins:02d}:{secs:02d}",
            "gen_time": round(gen_elapsed, 1),
            "recent_games": recent_games[-20:],
            "archetype_stats": _arch_stats,
            # V8: relic telemetry
            "relic_pool_size": relic_pool_size,
            "unique_relics_seen": len(relic_counts),
            "total_relics_picked": total_relics_seen,
            "avg_relics_per_run": round(total_relics_seen / max(1, total_games), 2),
            "top_relics": [
                {"id": _rid, "count": _cnt}
                for _rid, _cnt in relic_counts.most_common(20)
            ],
            "status": f"Gen {gen}/{num_generations} complete",
            "timestamp": time.time(),
        }
        _write_progress(progress_path, stats)

        # Console output (minimal for headless)
        win_pct = total_wins / max(1, total_games) * 100
        cur_lr = scheduler.get_last_lr()[0]
        print(
            f"Gen {gen:4d} | games={total_games} win={win_pct:.0f}% | "
            f"loss={total_loss:.3f} (v={v_loss:.3f} p={p_loss:.3f} o={o_loss:.3f}) | "
            f"lr={cur_lr:.1e} | {gen_elapsed:.1f}s",
            flush=True,
        )

        # Save checkpoint
        if gen % 10 == 0:
            ckpt_path = save_path / f"gen_{gen:04d}.pt"
            torch.save({
                "generation": gen,
                "model_state": network.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "games_played": total_games,
                "win_rate": total_wins / max(1, total_games),
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path.name}")

    print(f"Training complete! {total_games} games, {total_wins/max(1,total_games):.1%} win rate")


# ---------------------------------------------------------------------------
# Monitor: TUI dashboard (reads progress file)
# ---------------------------------------------------------------------------

def train_monitor(progress_file: str | None = None, refresh_rate: float = 1.0):
    """Live TUI dashboard that reads progress from the worker's JSON file."""
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    progress_path = Path(progress_file) if progress_file else _default_progress_path()
    console = Console()

    def build_dashboard(stats: dict) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="stats", ratio=1),
            Layout(name="games", ratio=1),
        )

        layout["header"].update(Panel(
            Text("STS2 AlphaZero Self-Play Training", style="bold cyan", justify="center"),
            style="cyan",
        ))

        # Stats
        st = Table(show_header=False, expand=True, box=None)
        st.add_column("Key", style="dim")
        st.add_column("Value", style="bold")
        st.add_row("Generation", f"{stats.get('generation', 0)}/{stats.get('num_generations', '?')}")
        st.add_row("Games Played", str(stats.get("games_played", 0)))
        st.add_row("Win Rate", f"{stats.get('win_rate', 0):.1%}")
        st.add_row("Gen Win Rate", f"{stats.get('gen_win_rate', 0):.1%}")
        st.add_row("Buffer Size", f"{stats.get('buffer_size', 0):,}")
        st.add_row("", "")
        st.add_row("Total Loss", f"{stats.get('total_loss', 0):.4f}")
        st.add_row("Value Loss", f"{stats.get('value_loss', 0):.4f}")
        st.add_row("Policy Loss", f"{stats.get('policy_loss', 0):.4f}")
        st.add_row("Option Loss", f"{stats.get('option_loss', 0):.4f}")
        st.add_row("", "")
        st.add_row("Buffers", f"combat={stats.get('buffer_size', 0):,}  option={stats.get('option_buffer_size', 0):,}")
        st.add_row("Learning Rate", f"{stats.get('lr', 0):.1e}")
        st.add_row("Sims/Move", str(stats.get("mcts_sims", "?")))
        st.add_row("Gen Time", f"{stats.get('gen_time', 0):.1f}s")
        st.add_row("Elapsed", stats.get("elapsed", "0:00"))
        layout["stats"].update(Panel(st, title="Training Stats"))

        # Recent games
        gt = Table(expand=True, box=None)
        gt.add_column("#", style="dim", width=4)
        gt.add_column("Combats", width=20)
        gt.add_column("Result", width=6)
        gt.add_column("Floor", width=5)
        gt.add_column("HP", width=4)
        for game in stats.get("recent_games", [])[-15:]:
            style = "green" if game["outcome"] == "win" else "red"
            enc = game.get("encounter", "?")
            # Support both old "turns" key and new "floor" key
            floor = game.get("floor", game.get("turns", "?"))
            gt.add_row(
                str(game["num"]),
                enc[:20],
                Text(game["outcome"], style=style),
                str(floor),
                str(game.get("hp", "?")),
            )
        layout["games"].update(Panel(gt, title="Recent Games"))

        layout["footer"].update(Panel(
            Text(stats.get("status", "Waiting for worker..."), justify="center"),
            style="dim",
        ))
        return layout

    console.print(f"[dim]Watching: {progress_path}[/dim]")
    console.print("[dim]Press Ctrl+C to stop (worker continues running)[/dim]\n")

    with Live(build_dashboard({}), console=console, refresh_per_second=refresh_rate) as live:
        try:
            while True:
                stats = _read_progress(progress_path)
                live.update(build_dashboard(stats))
                time.sleep(1.0 / refresh_rate)
        except KeyboardInterrupt:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Enable faulthandler so segfaults write a Python traceback to a file
    import faulthandler
    _fault_file = open(str(Path(__file__).resolve().parents[4] / "segfault_trace.log"), "w")
    faulthandler.enable(file=_fault_file)

    parser = argparse.ArgumentParser(description="STS2 AlphaZero Self-Play")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Run headless training worker")
    train_parser.add_argument("--generations", type=int, default=100)
    train_parser.add_argument("--games-per-gen", type=int, default=7)
    train_parser.add_argument("--sims", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=64)
    train_parser.add_argument("--epochs", type=int, default=3)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--temperature", type=float, default=1.0)
    train_parser.add_argument("--save-dir", type=str, default=None)
    train_parser.add_argument("--progress-file", type=str, default=None)
    train_parser.add_argument("--boss-log-file", type=str, default=None,
                              help="Where to append boss-fight detail JSONL "
                                   "(default: <save-dir>/boss_fights.jsonl)")

    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Live TUI dashboard")
    monitor_parser.add_argument("--progress-file", type=str, default=None)
    monitor_parser.add_argument("--refresh", type=float, default=1.0)

    args = parser.parse_args()

    if args.command == "train":
        train_worker(
            num_generations=args.generations,
            games_per_generation=args.games_per_gen,
            mcts_simulations=args.sims,
            batch_size=args.batch_size,
            train_epochs=args.epochs,
            lr=args.lr,
            temperature=args.temperature,
            save_dir=args.save_dir,
            progress_file=args.progress_file,
            boss_log_file=args.boss_log_file,
        )
    elif args.command == "monitor":
        train_monitor(
            progress_file=args.progress_file,
            refresh_rate=args.refresh,
        )
