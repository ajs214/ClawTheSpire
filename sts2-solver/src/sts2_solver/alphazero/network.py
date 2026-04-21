"""AlphaZero neural network for STS2 combat.

Architecture:
    State encoder (shared trunk):
        - Card embeddings (learned, 32-dim per card ID)
        - Hand: card embed (32) + stats (15) → self-attention → mean pool → 32-dim
        - Piles (draw/discard/exhaust): mean card embeddings → project → 32-dim each
        - Player: scalar features (HP, block, energy) + power embeddings
        - Enemies: per-slot features → linear projection → 32-dim × max_enemies
        - Relics: mean embeddings (8-dim)
        - Scalars: floor, turn, gold, deck_size, pending_choice, choice_type
        - Concatenated → MLP trunk (residual + LayerNorm) → 256-dim hidden state

    Value head:
        hidden → Linear(256→64) → ReLU → Linear(64→1) (unbounded, no tanh)

    Policy head (action embedding similarity):
        - Encode each legal action as: card_embed + features (target/flags)
        - Score = dot(hidden_projected, action_embed)
        - Supports play_card, end_turn, use_potion, and choose_card actions

    Option evaluation head (non-combat decisions except card picks):
        hidden + option_type_embed + card_embed → Linear(304→64) → ReLU → Linear(64→1)
        Handles rest/smith, map pathing, shop buy/remove/leave.
        Type embedding carries context (free reward vs gold cost vs removal).

    Card-pick evaluation head (deck-aware):
        hidden + deck_summary + card_embed + type_embed → 128 → 64 → 1
        Dedicated head for card reward picks with deck composition context.
        Deck summary is mean-pooled card embeddings → linear projection.
        Trained with MSE + ranking loss for better relative card evaluation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import (
    EncoderConfig,
    Vocabs,
    CARD_TYPE_MAP,
    TARGET_TYPE_MAP,
    PAD_IDX,
    RELIC_SYNERGY_DIM,
)

if TYPE_CHECKING:
    pass

# Mirror of self_play.OPTION_EVENT_CHOICE — duplicated here to avoid
# circular import (self_play imports STS2Network).
_OPTION_EVENT_CHOICE = 15


class CardSetEncoder(nn.Module):
    """Encode a variable-size set of cards using self-attention.

    Input: (batch, max_cards, card_feature_dim)
    Output: (batch, card_embed_dim)

    Uses one multi-head self-attention layer followed by mean pooling
    over non-padded positions.
    """

    def __init__(self, config: EncoderConfig):
        super().__init__()
        dim = config.card_feature_dim
        self.project_in = nn.Linear(dim, config.card_embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.card_embed_dim,
            num_heads=config.hand_attention_heads,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(config.card_embed_dim)
        self.card_embed_dim = config.card_embed_dim

    def forward(self, card_features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            card_features: (batch, max_cards, card_feature_dim)
            mask: (batch, max_cards) — True for padded positions
        Returns:
            (batch, card_embed_dim)
        """
        x = self.project_in(card_features)  # (batch, max_cards, embed_dim)

        # Handle empty sets (all masked) — return zeros to avoid attention NaN
        valid_mask = (~mask).unsqueeze(-1).float()  # (batch, max_cards, 1)
        num_valid = valid_mask.sum(dim=1)  # (batch, 1)
        if (num_valid == 0).all():
            return torch.zeros(x.shape[0], self.card_embed_dim, device=x.device)

        # Self-attention with padding mask
        attn_out, _ = self.attention(x, x, x, key_padding_mask=mask)
        x = self.layer_norm(x + attn_out)  # Residual + norm

        # Mean pool over non-padded positions
        pooled = (x * valid_mask).sum(dim=1) / num_valid.clamp(min=1)
        return pooled  # (batch, card_embed_dim)


class STS2Network(nn.Module):
    """AlphaZero-style network for STS2 combat.

    Takes encoded state tensors and produces:
        - value: scalar win probability in [-1, 1]
        - policy: scores for each legal action (pre-softmax logits)
    """

    def __init__(self, vocabs: Vocabs, config: EncoderConfig | None = None):
        super().__init__()
        self.config = config or EncoderConfig()
        self.vocabs = vocabs
        cfg = self.config

        # --- Embedding tables ---
        self.card_embed = nn.Embedding(
            len(vocabs.cards), cfg.card_embed_dim, padding_idx=PAD_IDX
        )
        self.relic_embed = nn.Embedding(
            len(vocabs.relics), cfg.relic_embed_dim, padding_idx=PAD_IDX
        )
        self.intent_embed = nn.Embedding(
            len(vocabs.intent_types), cfg.intent_embed_dim, padding_idx=PAD_IDX
        )
        self.power_embed = nn.Embedding(
            len(vocabs.powers), cfg.power_embed_dim, padding_idx=PAD_IDX
        )

        # --- Hand encoder (set attention) ---
        self.hand_encoder = CardSetEncoder(cfg)

        # --- Pile encoders (simple linear from summed embeddings) ---
        self.pile_project = nn.Linear(cfg.card_embed_dim, cfg.pile_feature_dim)

        # --- Enemy encoder ---
        self.enemy_project = nn.Linear(cfg.enemy_feature_dim, cfg.enemy_projected_dim)

        # --- Trunk MLP ---
        trunk_input_dim = cfg.state_dim
        # Trunk with residual connection + layer norm for stable training
        self.trunk_in = nn.Linear(trunk_input_dim, 256)
        self.trunk_hidden = nn.Linear(256, 256)
        self.trunk_norm = nn.LayerNorm(256)
        self.trunk_dropout = nn.Dropout(0.1)

        # --- Value head ---
        # Linear output (no Tanh) — targets are clamped to [-1, 1] by
        # _assign_run_values. Avoids Tanh gradient saturation near ±1.
        self.value_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # --- Policy head ---
        # Actions are encoded as: learned card embedding + feature vector (target/flags)
        # action_feat_dim = max_enemies+1 (target) + 5 (potion type) + 3 (flags: end_turn, use_potion, choose_card)
        action_feat_dim = cfg.action_feat_dim
        policy_action_dim = cfg.card_embed_dim + action_feat_dim
        self.policy_project = nn.Linear(256, policy_action_dim)
        self.action_project = nn.Linear(policy_action_dim, policy_action_dim)

        # --- Option evaluation head ---
        # Unified head for ALL non-combat decisions: card rewards, rest/smith,
        # map pathing, and shop (buy/remove/leave). Option type embedding
        # carries context (free reward vs 75g purchase vs 50g removal).
        self.option_type_embed = nn.Embedding(cfg.num_option_types, cfg.option_type_embed_dim, padding_idx=0)
        # V10: dedicated embedding for event-choice options, replacing the
        # positional placeholder that abused card_embed indices.
        self.event_choice_embed = nn.Embedding(
            cfg.num_event_choices, cfg.event_choice_embed_dim, padding_idx=0)
        self.option_eval_head = nn.Sequential(
            nn.Linear(256 + cfg.option_type_embed_dim + cfg.card_embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

        # --- Dedicated card-pick evaluation head ---
        # Richer than the generic option_eval_head: incorporates a learned
        # deck composition summary so the network can reason about synergy,
        # curve, and bloat when choosing which card to add (or skip).
        #
        # Input: hidden(256) + deck_summary(32) + relic_embed(8) + synergy_features(13) + card_embed(32) + type_embed(16) = 357
        # Two hidden layers (128→64) for modelling deck×card×relic interactions.
        self.deck_summary_project = nn.Linear(cfg.card_embed_dim, cfg.card_embed_dim)
        _card_head_input_dim = 256 + cfg.card_embed_dim + cfg.relic_embed_dim + RELIC_SYNERGY_DIM + cfg.card_embed_dim + cfg.option_type_embed_dim  # 357
        self.card_eval_head = nn.Sequential(
            nn.Linear(_card_head_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_state(
        self,
        hand_features: torch.Tensor,    # (batch, max_hand, card_feature_dim)
        hand_mask: torch.Tensor,         # (batch, max_hand) — True = padded
        hand_card_ids: torch.Tensor,     # (batch, max_hand) — card vocab indices
        draw_card_ids: torch.Tensor,     # (batch, max_draw) — card vocab indices
        draw_mask: torch.Tensor,         # (batch, max_draw)
        discard_card_ids: torch.Tensor,  # (batch, max_discard)
        discard_mask: torch.Tensor,      # (batch, max_discard)
        exhaust_card_ids: torch.Tensor,  # (batch, max_exhaust)
        exhaust_mask: torch.Tensor,      # (batch, max_exhaust)
        player_scalars: torch.Tensor,    # (batch, 5)
        player_power_ids: torch.Tensor,  # (batch, max_player_powers)
        player_power_amts: torch.Tensor, # (batch, max_player_powers)
        enemy_scalars: torch.Tensor,     # (batch, max_enemies, 6)
        enemy_power_ids: torch.Tensor,   # (batch, max_enemies * max_enemy_powers)
        enemy_power_amts: torch.Tensor,  # (batch, max_enemies * max_enemy_powers)
        relic_ids: torch.Tensor,         # (batch, max_relics)
        relic_mask: torch.Tensor,        # (batch, max_relics)
        potion_features: torch.Tensor,   # (batch, max_potions * potion_feature_dim)
        scalars: torch.Tensor,           # (batch, 4) — floor, turn, gold, deck_size
    ) -> torch.Tensor:
        """Encode full state into a hidden vector. Returns (batch, 256)."""
        batch = hand_features.shape[0]
        cfg = self.config

        # Hand: card embeddings concatenated with stats → attention → pool
        hand_embeds = self.card_embed(hand_card_ids)  # (batch, max_hand, 32)
        hand_input = torch.cat([hand_embeds, hand_features], dim=-1)  # + stats
        hand_vec = self.hand_encoder(hand_input, hand_mask)  # (batch, 32)

        # Piles: mean card embeddings, project (#7 — mean preserves scale across pile sizes)
        def encode_pile(card_ids, mask):
            embeds = self.card_embed(card_ids)  # (batch, max_pile, 32)
            valid = (~mask).unsqueeze(-1).float()
            count = valid.sum(dim=1).clamp(min=1)  # (batch, 1)
            meaned = (embeds * valid).sum(dim=1) / count  # (batch, 32)
            return self.pile_project(meaned)

        draw_vec = encode_pile(draw_card_ids, draw_mask)
        discard_vec = encode_pile(discard_card_ids, discard_mask)
        exhaust_vec = encode_pile(exhaust_card_ids, exhaust_mask)

        # Player: scalars + power embeddings concatenated with amounts
        p_pow_embeds = self.power_embed(player_power_ids)  # (batch, max_pp, 8)
        p_pow_amts = player_power_amts.unsqueeze(-1)       # (batch, max_pp, 1)
        p_pow_combined = torch.cat([p_pow_embeds, p_pow_amts], dim=-1)  # (batch, max_pp, 9)
        p_pow_flat = p_pow_combined.reshape(batch, -1)      # (batch, max_pp * 9)
        player_features = torch.cat([player_scalars, p_pow_flat], dim=-1)

        # Enemies: scalars + power embeddings per enemy slot
        total_e_powers = cfg.max_enemies * cfg.max_enemy_powers
        e_pow_embeds = self.power_embed(enemy_power_ids)    # (batch, total, 8)
        e_pow_amts = enemy_power_amts.unsqueeze(-1)         # (batch, total, 1)
        e_pow_combined = torch.cat([e_pow_embeds, e_pow_amts], dim=-1)  # (batch, total, 9)
        e_pow_flat = e_pow_combined.reshape(batch, cfg.max_enemies, cfg.max_enemy_powers * (cfg.power_embed_dim + 1))

        # Concatenate enemy scalars with their power features
        enemy_full = torch.cat([enemy_scalars, e_pow_flat], dim=-1)  # (batch, max_enemies, enemy_feature_dim)
        enemy_vecs = self.enemy_project(enemy_full)         # (batch, max_enemies, enemy_projected_dim)
        enemy_flat = enemy_vecs.reshape(batch, cfg.max_enemies * cfg.enemy_projected_dim)

        # Relics: mean embeddings (#6 — normalize by count)
        relic_embeds = self.relic_embed(relic_ids)  # (batch, max_relics, 8)
        relic_valid = (~relic_mask).unsqueeze(-1).float()
        relic_count = relic_valid.sum(dim=1).clamp(min=1)  # (batch, 1)
        relic_vec = (relic_embeds * relic_valid).sum(dim=1) / relic_count  # (batch, 8)

        # Concatenate everything
        state_vec = torch.cat([
            hand_vec, draw_vec, discard_vec, exhaust_vec,
            player_features, enemy_flat, relic_vec, potion_features, scalars,
        ], dim=-1)

        # Trunk with residual + layer norm
        h = F.relu(self.trunk_in(state_vec))                     # (batch, 256)
        h = h + self.trunk_dropout(F.relu(self.trunk_hidden(h))) # residual + dropout on output
        h = self.trunk_norm(h)
        return h

    def forward(
        self,
        hidden: torch.Tensor,            # (batch, 256)
        action_card_ids: torch.Tensor,    # (batch, max_actions) — card vocab indices
        action_features: torch.Tensor,    # (batch, max_actions, action_feat_dim)
        action_mask: torch.Tensor,        # (batch, max_actions) — True = invalid
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            value: (batch, 1) — estimated run value
            policy_logits: (batch, max_actions) — masked logits
        """
        # Value
        value = self.value_head(hidden)

        # Policy: combine learned card embeddings with action features
        card_embeds = self.card_embed(action_card_ids)  # (batch, max_actions, card_embed_dim)
        action_combined = torch.cat([card_embeds, action_features], dim=-1)  # (batch, max_actions, policy_action_dim)

        state_action = self.policy_project(hidden)            # (batch, policy_action_dim)
        action_embeds = self.action_project(action_combined)  # (batch, max_actions, policy_action_dim)

        # Dot product: (batch, max_actions)
        logits = torch.einsum("bd,bnd->bn", state_action, action_embeds)

        # Mask invalid actions with large negative
        logits = logits.masked_fill(action_mask, float("-inf"))

        return value, logits

    def predict(
        self, hidden: torch.Tensor,
        action_card_ids: torch.Tensor,
        action_features: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> tuple[float, list[float]]:
        """Single-state inference for MCTS. Returns (value, policy_probs)."""
        with torch.no_grad():
            value, logits = self.forward(
                hidden.unsqueeze(0),
                action_card_ids.unsqueeze(0),
                action_features.unsqueeze(0),
                action_mask.unsqueeze(0),
            )
            probs = F.softmax(logits[0], dim=0)
            return value.item(), probs.tolist()

    # ------------------------------------------------------------------
    # Option evaluation (all non-combat decisions)
    # ------------------------------------------------------------------

    def evaluate_options(
        self,
        hidden: torch.Tensor,         # (batch, 256)
        option_types: torch.Tensor,    # (batch, num_options) — option type indices
        option_cards: torch.Tensor,    # (batch, num_options) — card/event-choice vocab indices
        option_mask: torch.Tensor,     # (batch, num_options) — True = invalid/padded
    ) -> torch.Tensor:
        """Score a set of discrete options. Returns (batch, num_options) scores (unbounded)."""
        type_embeds = self.option_type_embed(option_types)      # (B, N, 16)

        # V10: event-choice options use a dedicated embedding table instead
        # of the card embedding.  We compute both lookups and select per-slot
        # based on whether option_type == OPTION_EVENT_CHOICE.
        is_event = (option_types == _OPTION_EVENT_CHOICE)       # (B, N)
        # Clamp indices to valid range for each table to avoid OOB
        card_idx = option_cards.clamp(0, self.card_embed.num_embeddings - 1)
        evt_idx = option_cards.clamp(0, self.event_choice_embed.num_embeddings - 1)
        card_embeds = self.card_embed(card_idx)                 # (B, N, 32)
        evt_embeds = self.event_choice_embed(evt_idx)           # (B, N, 32)
        opt_embeds = torch.where(
            is_event.unsqueeze(-1), evt_embeds, card_embeds)    # (B, N, 32)

        batch, num_opts, _ = type_embeds.shape
        hidden_exp = hidden.unsqueeze(1).expand(-1, num_opts, -1)  # (B, N, 256)

        combined = torch.cat([hidden_exp, type_embeds, opt_embeds], dim=-1)  # (B, N, 304)
        scores = self.option_eval_head(combined).squeeze(-1)      # (B, N)

        # Mask invalid options with large negative
        scores = scores.masked_fill(option_mask, -1e9)
        return scores

    def pick_best_option(
        self,
        hidden: torch.Tensor,       # (1, 256)
        option_types: list[int],
        option_cards: list[int],
    ) -> tuple[int, list[float]]:
        """Pick the highest-scoring option. Returns (best_index, all_scores)."""
        with torch.no_grad():
            device = hidden.device
            types_t = torch.tensor([option_types], dtype=torch.long, device=device)
            cards_t = torch.tensor([option_cards], dtype=torch.long, device=device)
            mask = torch.zeros(1, len(option_types), dtype=torch.bool, device=device)
            scores = self.evaluate_options(hidden, types_t, cards_t, mask)
            scores_list = scores[0].tolist()
            best_idx = max(range(len(scores_list)), key=lambda i: scores_list[i])
            return best_idx, scores_list

    # ------------------------------------------------------------------
    # Dedicated card-pick evaluation (deck-aware)
    # ------------------------------------------------------------------

    def _encode_deck_summary(
        self,
        deck_card_ids: torch.Tensor,  # (batch, max_deck)
        deck_mask: torch.Tensor,      # (batch, max_deck) — True = padded
    ) -> torch.Tensor:
        """Mean-pool deck card embeddings → project → (batch, card_embed_dim)."""
        embeds = self.card_embed(deck_card_ids)            # (B, D, 32)
        valid = (~deck_mask).unsqueeze(-1).float()         # (B, D, 1)
        count = valid.sum(dim=1).clamp(min=1)              # (B, 1)
        meaned = (embeds * valid).sum(dim=1) / count       # (B, 32)
        return self.deck_summary_project(meaned)           # (B, 32)

    def evaluate_card_picks(
        self,
        hidden: torch.Tensor,          # (batch, 256)
        deck_card_ids: torch.Tensor,    # (batch, max_deck)
        deck_mask: torch.Tensor,        # (batch, max_deck)
        option_types: torch.Tensor,     # (batch, num_options)
        option_cards: torch.Tensor,     # (batch, num_options)
        option_mask: torch.Tensor,      # (batch, num_options)
        relic_ids: torch.Tensor | None = None,    # (batch, max_relics)
        relic_mask: torch.Tensor | None = None,   # (batch, max_relics)
        synergy_features: torch.Tensor | None = None,  # (batch, RELIC_SYNERGY_DIM)
    ) -> torch.Tensor:
        """Score card-pick options with deck-composition and relic-synergy awareness.

        NOTE: Callers should pass relic_ids/relic_mask/synergy_features for
        relic-aware card evaluation. If omitted, zeros are used (backward compatible).
        See encoding.compute_relic_synergy_features() for the synergy vector.

        Returns (batch, num_options) unbounded scores.
        """
        deck_summary = self._encode_deck_summary(deck_card_ids, deck_mask)  # (B, 32)
        type_embeds = self.option_type_embed(option_types)                   # (B, N, 16)
        card_idx = option_cards.clamp(0, self.card_embed.num_embeddings - 1)
        card_embeds = self.card_embed(card_idx)                              # (B, N, 32)

        batch, num_opts, _ = type_embeds.shape
        hidden_exp = hidden.unsqueeze(1).expand(-1, num_opts, -1)            # (B, N, 256)
        deck_exp = deck_summary.unsqueeze(1).expand(-1, num_opts, -1)        # (B, N, 32)

        # Relic context: mean-pooled relic embeddings + synergy features
        if relic_ids is not None and relic_mask is not None:
            r_embeds = self.relic_embed(relic_ids)
            r_valid = (~relic_mask).unsqueeze(-1).float()
            r_count = r_valid.sum(dim=1).clamp(min=1)
            relic_vec = (r_embeds * r_valid).sum(dim=1) / r_count  # (B, 8)
        else:
            relic_vec = torch.zeros(batch, self.config.relic_embed_dim, device=hidden.device)

        if synergy_features is None:
            synergy_features = torch.zeros(batch, RELIC_SYNERGY_DIM, device=hidden.device)

        relic_context = torch.cat([relic_vec, synergy_features], dim=-1)  # (B, 8+13=21)
        relic_exp = relic_context.unsqueeze(1).expand(-1, num_opts, -1)   # (B, N, 21)

        # [hidden(256) + deck_summary(32) + relic_context(21) + card_embed(32) + type_embed(16)] = 357
        combined = torch.cat([hidden_exp, deck_exp, relic_exp, card_embeds, type_embeds], dim=-1)
        scores = self.card_eval_head(combined).squeeze(-1)                   # (B, N)

        scores = scores.masked_fill(option_mask, -1e9)
        return scores

    def pick_best_card(
        self,
        hidden: torch.Tensor,
        deck_card_ids: list[int],
        option_types: list[int],
        option_cards: list[int],
        relic_ids: list[int] | None = None,
        relic_mask: list[bool] | None = None,
        synergy_features: list[float] | None = None,
    ) -> tuple[int, list[float]]:
        """Pick the highest-scoring card option (deck and relic-aware). Returns (best_index, all_scores)."""
        with torch.no_grad():
            device = hidden.device
            types_t = torch.tensor([option_types], dtype=torch.long, device=device)
            cards_t = torch.tensor([option_cards], dtype=torch.long, device=device)
            opt_mask = torch.zeros(1, len(option_types), dtype=torch.bool, device=device)

            # Build deck tensor with padding
            if deck_card_ids:
                deck_t = torch.tensor([deck_card_ids], dtype=torch.long, device=device)
                deck_mask = torch.zeros(1, len(deck_card_ids), dtype=torch.bool, device=device)
            else:
                deck_t = torch.zeros(1, 1, dtype=torch.long, device=device)
                deck_mask = torch.ones(1, 1, dtype=torch.bool, device=device)

            # Build relic tensors
            if relic_ids is not None:
                relic_t = torch.tensor([relic_ids], dtype=torch.long, device=device)
                rmask_t = torch.tensor([relic_mask], dtype=torch.bool, device=device) if relic_mask else torch.zeros(1, len(relic_ids), dtype=torch.bool, device=device)
            else:
                relic_t = None
                rmask_t = None

            syn_t = torch.tensor([synergy_features], dtype=torch.float32, device=device) if synergy_features else None

            scores = self.evaluate_card_picks(
                hidden, deck_t, deck_mask, types_t, cards_t, opt_mask,
                relic_ids=relic_t, relic_mask=rmask_t, synergy_features=syn_t)
            scores_list = scores[0].tolist()

            # Pick bonus: gently bias toward taking a card over skipping,
            # scaled by how many cards have been drafted.  Fades to zero
            # at 15+ drafted cards so the network can skip freely later.
            n_drafted = len(deck_card_ids)  # base cards already filtered out
            pick_bonus = max(0.0, 0.15 * (1.0 - n_drafted / 15.0))
            if pick_bonus > 0 and len(scores_list) >= 2:
                skip_idx = len(scores_list) - 1
                best_card_idx = max(range(skip_idx), key=lambda i: scores_list[i])
                scores_list[best_card_idx] += pick_bonus

            best_idx = max(range(len(scores_list)), key=lambda i: scores_list[i])
            return best_idx, scores_list

    @staticmethod
    def pad_card_eval_weights(old_state_dict: dict, new_input_dim: int, old_input_dim: int = 336) -> dict:
        """Pad card_eval_head weights to accommodate new relic features.

        Copies old weights and zero-initializes new feature columns,
        preserving all previously learned behavior.
        """
        import copy
        state = copy.deepcopy(old_state_dict)
        key = "card_eval_head.0.weight"  # First linear layer
        if key in state and state[key].shape[1] == old_input_dim:
            old_w = state[key]
            new_w = torch.zeros(old_w.shape[0], new_input_dim, device=old_w.device, dtype=old_w.dtype)
            new_w[:, :old_input_dim] = old_w
            state[key] = new_w
        return state
