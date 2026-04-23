"""Monte Carlo Tree Search for STS2 combat.

AlphaZero-style MCTS: uses a neural network for both value estimation
(no rollouts needed) and policy prior (guides exploration toward
promising moves). The search tree spans multiple turns, including
card plays, end-of-turn, and enemy phases.

Usage:
    mcts = MCTS(network, vocabs, config)
    action, policy, root_value, actions = mcts.search(state, num_simulations=100)

Each simulation:
    1. SELECT:   Walk tree via PUCT (balances exploitation + exploration)
    2. EXPAND:   At a leaf, query network for value and policy prior
    3. BACKUP:   Propagate value up the tree

The tree handles STS2's sequential card play naturally:
    - Each node is a CombatState
    - Actions are individual card plays OR end_turn
    - end_turn triggers enemy phase → new turn → new set of actions
    - Terminal nodes (combat won/lost) have fixed values

Dynamic simulation budget:
    scale_simulations(base, num_actions) adjusts simulation count based on
    decision complexity.  Simple decisions (≤3 actions) get fewer sims,
    complex ones (10+ actions) get more.
"""

from __future__ import annotations

import math
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from ..actions import Action, END_TURN, enumerate_actions
from ..combat_engine import is_combat_over
from ..models import CombatState
from ..sim_step import step

if TYPE_CHECKING:
    from ..data_loader import CardDB
    from .encoding import EncoderConfig, Vocabs
    from .network import STS2Network


# ---------------------------------------------------------------------------
# Dynamic simulation budget
# ---------------------------------------------------------------------------

def scale_simulations(base_sims: int, num_actions: int, *, is_boss: bool = False) -> int:
    """Scale MCTS simulations based on decision complexity and encounter type.

    Simple decisions (1-3 actions) need fewer sims — the answer is often
    obvious. Complex decisions (10+ actions) benefit from extra search
    to find the right card sequencing.

    Boss fights get a 1.5x multiplier on top of complexity scaling because
    they're the highest-leverage combats (determine win/loss for the run).

    Returns an adjusted simulation count, roughly in [base*0.25, base*3.0].
    """
    if num_actions <= 1:
        sims = max(10, base_sims // 4)      # trivial: ~25%
    elif num_actions <= 3:
        sims = max(25, base_sims // 2)      # simple: ~50%
    elif num_actions <= 6:
        sims = base_sims                     # normal: 100%
    elif num_actions <= 10:
        sims = int(base_sims * 1.4)          # complex: 140%
    else:
        sims = int(base_sims * 2.0)          # very complex: 200% (big hands need deep search)

    if is_boss:
        sims = int(sims * 1.5)              # boss: +50% on top of complexity scaling

    return sims


# ---------------------------------------------------------------------------
# Prior boosting for under-explored free / energy-positive plays
# ---------------------------------------------------------------------------

def _boost_free_card_priors(
    actions: list[Action],
    probs: list[float],
    state: CombatState,
) -> list[float]:
    """Boost network priors for playable cards and dampen premature END_TURN.

    The neural network may undervalue card plays (especially early in
    training), causing MCTS to skip affordable plays and end turn with
    unspent energy.  We give multiplicative boosts so the tree search at
    least *tries* these actions, while still deferring to the network for
    relative ordering.

    Returns a re-normalised probability list (same length as input).
    """
    if state is None:
        return probs

    hand = state.player.hand
    energy = state.player.energy
    boosted = list(probs)

    # Card types that should never be boosted (junk cards enemies add to
    # your deck — Slimed, Wound, Dazed, Burn, etc.)
    from ..constants import CardType
    _JUNK_TYPES = {CardType.STATUS, CardType.CURSE}

    has_free = False       # any 0-cost non-junk play
    has_affordable = False  # any non-junk play the player can afford

    for i, action in enumerate(actions):
        if i >= len(boosted):
            break
        if action.action_type != "play_card" or action.card_idx is None:
            continue
        if action.card_idx >= len(hand):
            continue
        card = hand[action.card_idx]

        # Never boost Status/Curse cards — they're junk even if playable
        if card.card_type in _JUNK_TYPES:
            continue

        is_affordable = card.cost <= energy

        if card.cost == 0:
            has_free = True
        if is_affordable:
            has_affordable = True

        # 0-cost cards: always free to play, no reason to skip
        if card.cost == 0:
            boosted[i] *= 4.0
        # Energy-positive cards (e.g. Adrenaline): playing them opens options
        elif card.energy_gain and card.energy_gain > card.cost:
            boosted[i] *= 3.5
        # Cards that draw (e.g. Backflip, Acrobatics): expand options
        elif card.cards_draw and card.cards_draw >= 2:
            boosted[i] *= 2.5
        # Affordable cards the player can play right now
        elif is_affordable:
            boosted[i] *= 2.5

    # Dampen END_TURN when there are playable cards.
    # As the network improves from training with wasted-energy penalties,
    # these boosts will matter less — but keep them moderate for now.
    if has_free or has_affordable:
        dampen = 0.15 if has_free else 0.2
        for i, action in enumerate(actions):
            if i >= len(boosted):
                break
            if action.action_type == "end_turn":
                boosted[i] *= dampen

    # Re-normalise
    total = sum(boosted)
    if total > 0:
        return [p / total for p in boosted]
    return probs


# ---------------------------------------------------------------------------
# Transposition table — state hashing
# ---------------------------------------------------------------------------

def _hash_state(state: CombatState) -> int | None:
    """Compute a hash of the combat state for transposition detection.

    Two states that are strategically identical (same hand contents, HP,
    energy, enemy states, etc.) map to the same hash.  Returns None if
    the state has a pending choice (those are rare and order-sensitive).
    """
    if state.pending_choice is not None:
        return None

    p = state.player
    try:
        # Hand as a sorted tuple of card IDs (order doesn't matter)
        hand_key = tuple(sorted(c.id for c in p.hand))

        # Enemy state: (id, hp, block, powers_tuple) per enemy
        enemy_key = tuple(
            (e.id, e.hp, e.block, tuple(sorted(e.powers.items())))
            for e in state.enemies if e.is_alive
        )

        key = (
            p.hp, p.block, p.energy,
            tuple(sorted(p.powers.items())),
            hand_key,
            enemy_key,
            state.turn,
            state.cards_played_this_turn,
        )
        return hash(key)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class Node:
    """A node in the MCTS search tree."""

    state: CombatState
    parent: Node | None = None
    parent_action: Action | None = None

    # Tree statistics
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0  # Policy prior from network

    # Children: action → Node
    children: dict[int, Node] = field(default_factory=dict)
    # Legal actions at this node (populated on expansion)
    legal_actions: list[Action] = field(default_factory=list)
    is_expanded: bool = False
    is_terminal: bool = False
    terminal_value: float = 0.0

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float = 1.5) -> float:
        """Upper Confidence Bound for Trees (PUCT) score."""
        if self.visit_count == 0:
            # Unvisited: high exploration bonus
            return c_puct * self.prior * math.sqrt(parent_visits + 1)

        exploitation = self.value
        exploration = c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        return exploitation + exploration


# ---------------------------------------------------------------------------
# MCTS
# ---------------------------------------------------------------------------

class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""

    def __init__(
        self,
        network: STS2Network,
        vocabs: Vocabs,
        config: EncoderConfig | None = None,
        card_db: CardDB | None = None,
        c_puct: float = 1.5,
        device: str = "cpu",
    ):
        self.network = network
        self.vocabs = vocabs
        self.config = config
        self.card_db = card_db
        self.c_puct = c_puct
        self.device = device
        self.network.to(device)
        self.network.eval()

        # Transposition table: state_hash → value (float)
        # Cleared at the start of each search() call.
        self._transposition: dict[int, float] = {}

    def search(
        self,
        state: CombatState,
        num_simulations: int = 100,
        temperature: float = 1.0,
        time_limit_ms: float | None = None,
    ) -> tuple[Action, list[float], float, list[Action]]:
        """Run MCTS from the given state.

        Returns:
            action: The selected action
            policy: Visit-count-based policy distribution over legal actions
            root_value: Mean backed-up value at the root (win expectancy)
        """
        self._transposition.clear()
        root = Node(state=deepcopy(state))
        self._expand(root)

        if root.is_terminal or not root.legal_actions:
            return END_TURN, [1.0], root.terminal_value if root.is_terminal else 0.0, [END_TURN]

        deadline = None
        if time_limit_ms is not None:
            deadline = time.perf_counter() + time_limit_ms / 1000

        for _ in range(num_simulations):
            if deadline and time.perf_counter() > deadline:
                break

            # SELECT: walk tree to a leaf
            node = self._select(root)

            # EXPAND + EVALUATE
            if not node.is_terminal:
                value = self._expand(node)
            else:
                value = node.terminal_value

            # BACKUP: propagate value up
            self._backup(node, value)

        # Extract policy from visit counts
        actions = root.legal_actions
        visits = [
            root.children[i].visit_count if i in root.children else 0
            for i in range(len(actions))
        ]
        total_visits = sum(visits) or 1

        if temperature == 0:
            # Greedy: pick most visited
            best_idx = max(range(len(visits)), key=lambda i: visits[i])
            policy = [0.0] * len(actions)
            policy[best_idx] = 1.0
        else:
            # Temperature-scaled visit counts
            if temperature == 1.0:
                policy = [v / total_visits for v in visits]
            else:
                scaled = [v ** (1.0 / temperature) for v in visits]
                total_scaled = sum(scaled) or 1
                policy = [s / total_scaled for s in scaled]

        # Select action
        if temperature == 0:
            action_idx = max(range(len(visits)), key=lambda i: visits[i])
        else:
            # Sample from policy
            import random
            action_idx = random.choices(range(len(actions)), weights=policy, k=1)[0]

        return actions[action_idx], policy, root.value, actions

    def _select(self, root: Node) -> Node:
        """Walk tree from root to a leaf using PUCT selection."""
        node = root
        while node.is_expanded and not node.is_terminal:
            if not node.children:
                break
            # Pick child with highest UCB score
            best_idx = max(
                node.children.keys(),
                key=lambda i: node.children[i].ucb_score(node.visit_count, self.c_puct),
            )
            node = node.children[best_idx]
        return node

    def _expand(self, node: Node) -> float:
        """Expand a leaf node: get legal actions, query network for priors and value.

        Uses a transposition table keyed on a full state hash.  Because the
        hash captures HP, energy, block, powers, hand *contents* (sorted),
        and enemy states, two nodes only share a hash when the resulting
        game state is truly identical — regardless of how they got there.

        On a cache hit we still run the network for policy priors (so child
        nodes get proper exploration guidance) but substitute the cached
        value, which has been refined by prior search and is more accurate
        than a single network evaluation.

        Returns the value estimate for this state.
        """
        # Lazy state computation: compute on first visit
        if node.state is None and node.parent is not None and node.parent_action is not None:
            result = step(node.parent.state, node.parent_action, self.card_db)
            node.state = result.state
            if result.done:
                node.is_terminal = True
                node.terminal_value = 1.0 if result.outcome == "win" else -1.0
                node.is_expanded = True
                return node.terminal_value

        # Check for terminal state
        outcome = is_combat_over(node.state)
        if outcome is not None:
            node.is_terminal = True
            node.terminal_value = 1.0 if outcome == "win" else -1.0
            node.is_expanded = True
            return node.terminal_value

        # Get legal actions
        node.legal_actions = enumerate_actions(node.state)
        if not node.legal_actions:
            node.is_terminal = True
            node.terminal_value = 0.0
            node.is_expanded = True
            return 0.0

        # --- Transposition table lookup ---
        state_hash = _hash_state(node.state)
        cached_value = None
        if state_hash is not None and state_hash in self._transposition:
            cached_value = self._transposition[state_hash]

        # Query network (always needed for policy priors on child nodes)
        from .state_tensor import encode_state, encode_actions

        with torch.no_grad():
            state_tensors = encode_state(node.state, self.vocabs, self.config)
            state_tensors = {k: v.to(self.device) for k, v in state_tensors.items()}

            hidden = self.network.encode_state(**state_tensors)

            action_card_ids, action_features, action_mask = encode_actions(
                node.legal_actions, node.state, self.vocabs, self.config,
            )
            action_card_ids = action_card_ids.to(self.device)
            action_features = action_features.to(self.device)
            action_mask = action_mask.to(self.device)

            value, logits = self.network.forward(hidden, action_card_ids, action_features, action_mask)

            # Softmax over legal actions
            probs = torch.nn.functional.softmax(logits[0, :len(node.legal_actions)], dim=0)
            probs = probs.cpu().tolist()
            value = value.item()

        # On transposition hit, use the cached value (refined by prior
        # search) instead of the raw network estimate.
        if cached_value is not None:
            value = cached_value

        # Boost priors for free / energy-positive card plays so MCTS
        # actually explores them even when the network prior is weak.
        probs = _boost_free_card_priors(node.legal_actions, probs, node.state)

        # Create child nodes lazily — state computed on first visit
        for i, action in enumerate(node.legal_actions):
            child = Node(
                state=None,  # computed lazily on first visit
                parent=node,
                parent_action=action,
                prior=probs[i] if i < len(probs) else 1.0 / len(node.legal_actions),
            )
            node.children[i] = child

        node.is_expanded = True

        # Store in transposition table (first visit — raw network value)
        if state_hash is not None and cached_value is None:
            self._transposition[state_hash] = value

        return value

    def _backup(self, node: Node, value: float) -> None:
        """Propagate value up the tree to root."""
        current = node
        while current is not None:
            current.visit_count += 1
            current.value_sum += value
            current = current.parent

    def get_stats(self, root: Node) -> dict:
        """Get search statistics for debugging."""
        if not root.children:
            return {}
        actions = root.legal_actions
        stats = {}
        for i, action in enumerate(actions):
            if i in root.children:
                child = root.children[i]
                stats[str(action)] = {
                    "visits": child.visit_count,
                    "value": round(child.value, 3),
                    "prior": round(child.prior, 3),
                }
        return stats
