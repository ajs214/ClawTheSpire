"""Combat engine: turn lifecycle, card play, power ticks, enemy intents."""

from __future__ import annotations

import copy
import math
import random
from typing import TYPE_CHECKING

from .constants import CardType, TargetType
from .effects import (
    draw_cards,
    gain_block,
    calculate_block_gain,
    deal_damage,
    add_card_to_hand,
)
from .card_registry import get_effect
from .models import Card, CombatState, EnemyState
from . import relic_effects

if TYPE_CHECKING:
    from .data_loader import CardDB


# ---------------------------------------------------------------------------
# Card playability
# ---------------------------------------------------------------------------

def can_play_card(state: CombatState, card_idx: int) -> bool:
    """Check if a card in hand can be played."""
    if card_idx < 0 or card_idx >= len(state.player.hand):
        return False
    card = state.player.hand[card_idx]
    # Unplayable cards (Status, Curse) use cost -1 in game data
    if card.cost < 0:
        return False
    cost = effective_cost(state, card)
    if cost > state.player.energy:
        return False
    # Targeted cards need at least one living enemy
    if card.target in (TargetType.ANY_ENEMY, TargetType.RANDOM_ENEMY):
        if not any(e.is_alive for e in state.enemies):
            return False
    # Ringing: can only play 1 card this turn
    if state.player.powers.get("Ringing", 0) > 0 and state.cards_played_this_turn >= 1:
        return False
    # Velvet Choker: can only play 6 cards per turn
    if state.player.powers.get("Velvet Choker", 0) > 0 and state.cards_played_this_turn >= 6:
        return False
    # --- Conditional play restrictions ---
    # Grand Finale: can only be played if draw pile is empty
    if card.id == "GRAND_FINALE" and len(state.player.draw_pile) > 0:
        return False
    # Clash: can only be played if every card in hand is an Attack
    if card.id == "CLASH":
        for c in state.player.hand:
            if c.card_type != CardType.ATTACK:
                return False
    # Pact's End: can only be played if 3+ cards in exhaust pile
    if card.id == "PACTS_END" and len(state.player.exhaust_pile) < 3:
        return False
    return True


def effective_cost(state: CombatState, card: Card) -> int:
    """Get the effective energy cost of a card, accounting for powers."""
    cost = card.cost
    # Bullet Time: all cards free
    if state.player.all_cards_free:
        return 0
    # Brilliant Scarf: 5th card each turn is free
    if "BRILLIANT_SCARF" in state.relics and state.cards_played_this_turn >= 4:
        return 0
    # Corruption: Skills cost 0
    if card.card_type == CardType.SKILL and state.player.powers.get("Corruption", 0) > 0:
        return 0
    # X-cost cards spend all remaining energy
    if card.is_x_cost:
        return state.player.energy
    # Tangled: Attacks cost 1 more energy
    if card.card_type == CardType.ATTACK and state.player.powers.get("Tangled", 0) > 0:
        cost += 1
    return cost


def valid_targets(state: CombatState, card: Card) -> list[int]:
    """Return valid target indices for a card."""
    if card.target == TargetType.ANY_ENEMY:
        return [i for i, e in enumerate(state.enemies) if e.is_alive]
    if card.target == TargetType.RANDOM_ENEMY:
        return [i for i, e in enumerate(state.enemies) if e.is_alive]
    # Self, AllEnemies don't need a target
    return []


# ---------------------------------------------------------------------------
# Play a card
# ---------------------------------------------------------------------------

def play_card(
    state: CombatState,
    card_idx: int,
    target_idx: int | None = None,
    card_db: CardDB | None = None,
) -> None:
    """Play a card from hand. Mutates state in place.

    Args:
        state: Current combat state.
        card_idx: Index into player's hand.
        target_idx: Enemy index for targeted cards.
        card_db: Card database (needed for some custom effects).
    """
    card = state.player.hand[card_idx]
    cost = effective_cost(state, card)

    # Deduct energy (store X value for X-cost cards before deducting)
    if card.is_x_cost:
        state.last_x_cost = state.player.energy
        if "CHEMICAL_X" in state.relics:
            state.last_x_cost += 2
    state.player.energy -= cost

    # Remove from hand
    state.player.hand.pop(card_idx)

    # Track plays
    state.cards_played_this_turn += 1
    if card.card_type == CardType.ATTACK:
        state.attacks_played_this_turn += 1

    # --- Pre-effect triggers ---
    # Rage: gain block when playing an Attack
    if card.card_type == CardType.ATTACK:
        rage_amount = state.player.powers.get("Rage", 0)
        if rage_amount > 0:
            state.player.block += calculate_block_gain(rage_amount, state)

    # --- Execute card effect ---
    effect_fn = get_effect(card, card_db)
    effect_fn(state, target_idx)

    # THROWING_AXE: replay the effect once per turn
    if "THROWING_AXE" in state.relics and not state.player.powers.get("_throwing_axe_used"):
        state.player.powers["_throwing_axe_used"] = 1
        effect_fn(state, target_idx)

    # --- Burst: Skills played with active burst are played again ---
    # FIXED: If Burst is active and this is a Skill, apply its effect again
    if card.card_type == CardType.SKILL and state.player.burst_count > 0:
        effect_fn(state, target_idx)
        state.player.burst_count -= 1

    # --- Post-effect triggers ---
    # Dark Embrace: draw on exhaust (handled in _move_card_after_play)
    # Feel No Pain: block on exhaust (handled in _move_card_after_play)

    # Juggling: 3rd Attack each turn adds a copy to hand
    if (card.card_type == CardType.ATTACK
            and state.player.powers.get("Juggling", 0) > 0
            and state.attacks_played_this_turn == 3):
        state.player.hand.append(card)

    # --- Relic triggers on card play (dispatched through relic_effects) ---
    relic_effects.apply_card_play(state, card)

    # MUSIC_BOX: first attack each turn creates ethereal copy in discard
    if ("MUSIC_BOX" in state.relics and card.card_type == CardType.ATTACK
            and not state.player.powers.get("_music_box_used")):
        state.player.powers["_music_box_used"] = 1
        ethereal_copy = copy.copy(card)
        ethereal_copy.ethereal = True
        state.player.discard_pile.append(ethereal_copy)

    # --- Move card to appropriate zone ---
    _move_card_after_play(state, card)

    # HISTORY_COURSE: track last played attack/skill for replay next turn
    if "HISTORY_COURSE" in state.relics and card.card_type in (CardType.ATTACK, CardType.SKILL):
        state._history_course_last = card

    # UNCEASING_TOP: draw 1 if hand is empty after playing
    if "UNCEASING_TOP" in state.relics and not state.player.hand:
        draw_cards(state, 1)


def use_potion(state: CombatState, potion_idx: int) -> None:
    """Use a potion from the given slot. Mutates state in place.

    Supports all effect keys defined in simulator.POTION_TYPES:
      heal, block, energy, draw, strength, dexterity, damage_all,
      enemy_weak, enemy_vulnerable, enemy_poison, thorns, plated_armor,
      metallicize, ritual, regen, play_top_cards, gamblers_brew,
      duplication, fairy, smoke_bomb, fill_potions, max_hp, snecko_oil
    """
    if potion_idx >= len(state.player.potions):
        return
    pot = state.player.potions[potion_idx]
    if not pot:
        return

    # --- Direct healing ---
    if pot.get("heal"):
        state.player.hp = min(state.player.hp + pot["heal"], state.player.max_hp)

    # --- Block ---
    if pot.get("block"):
        state.player.block += pot["block"]

    # --- Energy ---
    if pot.get("energy"):
        state.player.energy += pot["energy"]

    # --- Draw cards ---
    if pot.get("draw"):
        draw_cards(state, pot["draw"])

    # --- Strength ---
    if pot.get("strength"):
        state.player.powers["Strength"] = (
            state.player.powers.get("Strength", 0) + pot["strength"]
        )

    # --- Dexterity ---
    if pot.get("dexterity"):
        state.player.powers["Dexterity"] = (
            state.player.powers.get("Dexterity", 0) + pot["dexterity"]
        )

    # --- AoE damage ---
    if pot.get("damage_all"):
        for e in state.enemies:
            if e.is_alive:
                dmg = pot["damage_all"]
                if e.block > 0:
                    if dmg >= e.block:
                        dmg -= e.block
                        e.block = 0
                    else:
                        e.block -= dmg
                        dmg = 0
                e.hp -= dmg

    # --- Enemy debuffs ---
    if pot.get("enemy_weak"):
        for e in state.enemies:
            if e.is_alive:
                e.powers["Weak"] = e.powers.get("Weak", 0) + pot["enemy_weak"]

    if pot.get("enemy_vulnerable"):
        for e in state.enemies:
            if e.is_alive:
                e.powers["Vulnerable"] = e.powers.get("Vulnerable", 0) + pot["enemy_vulnerable"]

    if pot.get("enemy_poison"):
        for e in state.enemies:
            if e.is_alive:
                e.powers["Poison"] = e.powers.get("Poison", 0) + pot["enemy_poison"]

    # --- Thorns (Liquid Bronze) ---
    if pot.get("thorns"):
        state.player.powers["Thorns"] = (
            state.player.powers.get("Thorns", 0) + pot["thorns"]
        )

    # --- Plated Armor (Essence of Steel): block each turn ---
    if pot.get("plated_armor"):
        state.player.powers["Metallicize"] = (
            state.player.powers.get("Metallicize", 0) + pot["plated_armor"]
        )

    # --- Metallicize (Heart of Iron): block each turn ---
    if pot.get("metallicize"):
        state.player.powers["Metallicize"] = (
            state.player.powers.get("Metallicize", 0) + pot["metallicize"]
        )

    # --- Ritual (Cultist Potion): +N Strength per turn ---
    if pot.get("ritual"):
        state.player.powers["Ritual"] = (
            state.player.powers.get("Ritual", 0) + pot["ritual"]
        )

    # --- Regen: heal N per turn for N turns ---
    if pot.get("regen"):
        state.player.powers["Regen"] = (
            state.player.powers.get("Regen", 0) + pot["regen"]
        )

    # --- Play top N cards from draw pile (Distilled Chaos) ---
    if pot.get("play_top_cards"):
        n = pot["play_top_cards"]
        for _ in range(n):
            if not state.player.draw_pile:
                break
            card = state.player.draw_pile.pop()
            alive = [i for i, e in enumerate(state.enemies) if e.is_alive]
            target = alive[0] if alive else None
            effect_fn = get_effect(card)
            effect_fn(state, target)
            state.player.discard_pile.append(card)

    # --- Gambler's Brew: discard hand, redraw same count ---
    if pot.get("gamblers_brew"):
        hand_size = len(state.player.hand)
        state.player.discard_pile.extend(state.player.hand)
        state.player.hand.clear()
        draw_cards(state, hand_size)

    # --- Duplication: next card plays twice (approximated as Burst 1) ---
    if pot.get("duplication"):
        state.player.burst_count = max(state.player.burst_count, 1)

    # --- Fairy in a Bottle: stored as a power, triggers on death ---
    if pot.get("fairy"):
        state.player.powers["Fairy"] = 1

    # --- Smoke Bomb: flee combat → modelled as big heal (avoids damage) ---
    if pot.get("smoke_bomb"):
        state.player.hp = min(state.player.hp + 20, state.player.max_hp)

    # --- Fruit Juice: permanent +5 max HP ---
    if pot.get("max_hp"):
        state.player.max_hp += pot["max_hp"]
        state.player.hp += pot["max_hp"]

    # --- Fill potions: modelled as heal + strength (proxy for value) ---
    if pot.get("fill_potions"):
        state.player.hp = min(state.player.hp + 10, state.player.max_hp)
        state.player.powers["Strength"] = (
            state.player.powers.get("Strength", 0) + 1
        )

    # REPTILE_TRINKET: gain +3 Strength when using a potion
    if "REPTILE_TRINKET" in state.relics:
        state.player.powers["Strength"] = state.player.powers.get("Strength", 0) + 3

    state.player.potions[potion_idx] = {}  # empty the slot


def _move_card_after_play(state: CombatState, card: Card) -> None:
    """Move a played card to the correct zone."""
    # RAZOR_TOOTH: upgrade card for remainder of combat after playing it
    if "RAZOR_TOOTH" in state.relics and not card.upgraded:
        card.upgraded = True
        # Bump stats approximately: +25% damage, +25% block
        if card.damage and card.damage > 0:
            card.damage = math.floor(card.damage * 1.25) + 1
        if card.block and card.block > 0:
            card.block = math.floor(card.block * 1.25) + 1

    should_exhaust = (
        card.exhausts
        or card.card_type == CardType.POWER
        or (card.card_type == CardType.SKILL
            and state.player.powers.get("Corruption", 0) > 0)
    )

    if should_exhaust:
        state.player.exhaust_pile.append(card)
        _on_exhaust(state)
    else:
        state.player.discard_pile.append(card)


def _on_exhaust(state: CombatState) -> None:
    """Trigger effects when a card is exhausted."""
    # Dark Embrace: draw 1 per stack
    dark_embrace = state.player.powers.get("Dark Embrace", 0)
    if dark_embrace > 0:
        draw_cards(state, dark_embrace)

    # Feel No Pain: gain block per stack
    fnp = state.player.powers.get("Feel No Pain", 0)
    if fnp > 0:
        state.player.block += calculate_block_gain(fnp, state)

    # BURNING_STICKS: first skill exhausted each combat → copy to hand
    if "BURNING_STICKS" in state.relics and not state.player.powers.get("_burning_sticks_used"):
        # Check if the exhausted card was a Skill
        if state.player.exhaust_pile and state.player.exhaust_pile[-1].card_type == CardType.SKILL:
            state.player.powers["_burning_sticks_used"] = 1
            card_copy = copy.copy(state.player.exhaust_pile[-1])
            state.player.hand.append(card_copy)


# ---------------------------------------------------------------------------
# Turn lifecycle
# ---------------------------------------------------------------------------

def start_combat(state: CombatState, is_elite: bool = False,
                  is_boss: bool = False) -> None:
    """Apply one-time start-of-combat relic effects. Call before first start_turn().

    Dispatches through ``relic_effects.apply_start_of_combat`` so the data
    tables in ``relic_effects.py`` are the single source of truth.
    """
    relic_effects.apply_start_of_combat(state, is_elite=is_elite, is_boss=is_boss)


def start_turn(state: CombatState) -> None:
    """Begin a new player turn. Mutates state in place."""
    state.turn += 1
    state.cards_played_this_turn = 0
    state.attacks_played_this_turn = 0
    state.discards_this_turn = 0  # Reset discard counter

    # Reset energy
    if "ICE_CREAM" in state.relics:
        state.player.energy = max(state.player.energy, 0) + state.player.max_energy
    else:
        state.player.energy = state.player.max_energy
    # Berserk: bonus energy
    berserk = state.player.powers.get("Berserk", 0)
    if berserk > 0:
        state.player.energy += berserk
    # PAELS_TEARS: +2 energy if had unspent energy last turn
    if state.player.powers.pop("_paels_tears_bonus", 0):
        state.player.energy += 2

    # Remove block (unless Barricade)
    if state.player.powers.get("Barricade", 0) <= 0:
        if "STURDY_CLAMP" in state.relics:
            state.player.block = min(state.player.block, 10)
        else:
            state.player.block = 0

    # Remove enemy block
    for enemy in state.enemies:
        enemy.block = 0

    # Start-of-turn power ticks
    _tick_start_of_turn_powers(state)

    # --- Per-turn counter resets for CARD_PLAY_TRIGGERS with scope="turn" ---
    # Any counter key in state.player.powers that starts with '_' and ends
    # with '_count' should be scanned to see if its owning relic rule is
    # turn-scoped; for simplicity we just clear the well-known turn-scoped ones.
    for key in (
        "_KUNAI_count", "_ORNAMENTAL_FAN_count", "_SHURIKEN_count",
        "_LETTER_OPENER_count", "_KUSARIGAMA_count", "_RAINBOW_RING_count",
        "_kunai_count", "_fan_count", "_shuriken_count", "_letter_opener_count",
        "_rainbow_attack", "_rainbow_skill", "_rainbow_power",
        "_music_box_used", "_damage_this_turn", "_throwing_axe_used",
    ):
        state.player.powers.pop(key, None)

    # --- Start-of-turn relic effects (dispatched through relic_effects) ---
    # Art of War: if no attacks last turn, +1 energy (tracked via power).
    # Kept inline because the "_art_of_war_eligible" flag spans turns and
    # is simpler than plumbing through a marker dispatch.
    if "ART_OF_WAR" in state.relics and state.turn > 1:
        if state.player.powers.get("_art_of_war_eligible", 0) > 0:
            state.player.energy += 1
        state.player.powers.pop("_art_of_war_eligible", None)

    relic_effects.apply_turn_start(state)

    # RED_SKULL: +3 Strength when at half health or lower
    if "RED_SKULL" in state.relics:
        is_low = state.player.hp <= state.player.max_hp // 2
        was_active = state.player.powers.get("_red_skull_active", 0)
        if is_low and not was_active:
            state.player.powers["Strength"] = state.player.powers.get("Strength", 0) + 3
            state.player.powers["_red_skull_active"] = 1
        elif not is_low and was_active:
            state.player.powers["Strength"] = state.player.powers.get("Strength", 0) - 3
            state.player.powers["_red_skull_active"] = 0

    # MR_STRUGGLES: deal turn-number damage to all enemies
    if "MR_STRUGGLES" in state.relics:
        for e in state.enemies:
            if e.is_alive:
                e.hp -= state.turn

    # POLLINOUS_CORE: every 4 turns, draw 2
    if "POLLINOUS_CORE" in state.relics and state.turn % 4 == 0:
        draw_cards(state, 2)

    # HAPPY_FLOWER: every 3 turns, +1 energy
    if "HAPPY_FLOWER" in state.relics and state.turn % 3 == 0:
        state.player.energy += 1

    # FAKE_HAPPY_FLOWER: every 5 turns, +1 energy
    if "FAKE_HAPPY_FLOWER" in state.relics and state.turn % 5 == 0:
        state.player.energy += 1

    # CROSSBOW: add a random attack from draw pile to hand, free this turn
    if "CROSSBOW" in state.relics and state.player.draw_pile:
        attacks = [c for c in state.player.draw_pile if c.card_type == CardType.ATTACK]
        if attacks:
            chosen = random.choice(attacks)
            state.player.draw_pile.remove(chosen)
            chosen_copy = copy.copy(chosen)
            chosen_copy.cost = 0
            chosen_copy.ethereal = True
            state.player.hand.append(chosen_copy)

    # HISTORY_COURSE: replay last played attack/skill
    if "HISTORY_COURSE" in state.relics and hasattr(state, '_history_course_last'):
        last = state._history_course_last
        if last is not None:
            alive = [i for i, e in enumerate(state.enemies) if e.is_alive]
            target = alive[0] if alive else None
            try:
                eff = get_effect(last)
                eff(state, target)
            except Exception:
                pass

    # Clear turn-duration powers from previous turn
    for power_name in ("Rage", "OneTwoPunch"):
        state.player.powers.pop(power_name, None)

    # Unmovable resets each turn
    if "Unmovable" in state.player.powers:
        state.player.powers["Unmovable_used"] = 0

    # Draw cards
    draw_cards(state, 5)


def end_turn(state: CombatState) -> None:
    """End the player's turn. Mutates state in place.

    Does NOT resolve enemy intents — call resolve_enemy_intents() separately
    so the solver can evaluate state before and after enemy actions.
    """
    # Stampede: play attack(s) from hand against random alive enemy (before discard)
    stampede = state.player.powers.get("Stampede", 0)
    for _ in range(stampede):
        attacks = [c for c in state.player.hand if c.card_type == CardType.ATTACK]
        if not attacks:
            break
        alive = [i for i, e in enumerate(state.enemies) if e.is_alive]
        if not alive:
            break
        card = attacks[0]  # deterministic for solver (first attack in hand)
        card_idx = state.player.hand.index(card)
        effect_fn = get_effect(card)
        state.player.hand.pop(card_idx)
        # FIXED: Use random.choice instead of deterministic alive[0]
        effect_fn(state, random.choice(alive))
        state.player.discard_pile.append(card)

    # --- End-of-turn relic effects (dispatched through relic_effects) ---
    relic_effects.apply_end_of_turn(state)

    # PAELS_TEARS: unspent energy → +2 next turn
    if "PAELS_TEARS" in state.relics and state.player.energy > 0:
        state.player.powers["_paels_tears_bonus"] = 1

    # DIAMOND_DIADEM: ≤2 cards played → half incoming damage next turn
    if "DIAMOND_DIADEM" in state.relics and state.cards_played_this_turn <= 2:
        state.player.powers["_diamond_diadem_active"] = 1
    else:
        state.player.powers.pop("_diamond_diadem_active", None)

    # Art of War: track if no attacks were played (checked next start_turn).
    # Cross-turn state so kept inline rather than in the data table.
    if "ART_OF_WAR" in state.relics:
        if state.attacks_played_this_turn == 0:
            state.player.powers["_art_of_war_eligible"] = 1

    # Infection cards: deal 3 damage per Infection in hand at end of turn
    for card in state.player.hand:
        if card.name == "Infection" or card.id == "INFECTION":
            state.player.hp -= 3

    # Constrict: deal damage equal to stacks at end of turn
    constrict = state.player.powers.get("Constrict", 0)
    if constrict > 0:
        state.player.hp -= constrict

    # Discard hand (except Retain)
    if "RUNIC_PYRAMID" in state.relics:
        # Hand persists — only exhaust Ethereal cards
        remaining = []
        for card in state.player.hand:
            if card.ethereal:
                state.player.exhaust_pile.append(card)
                _on_exhaust(state)
            else:
                remaining.append(card)
        state.player.hand = remaining
    else:
        # Normal discard
        remaining = []
        for card in state.player.hand:
            if card.retain:
                remaining.append(card)
            elif card.ethereal:
                state.player.exhaust_pile.append(card)
                _on_exhaust(state)
            else:
                state.player.discard_pile.append(card)
                state.discards_this_turn += 1  # Track for Memento Mori
        state.player.hand = remaining

    # End-of-turn power ticks
    _tick_end_of_turn_powers(state)


def resolve_enemy_intents(state: CombatState) -> None:
    """Resolve all enemy intents (attacks, buffs, debuffs, etc.).

    Handles:
    - Attack: deal damage to player (with all modifier calculations)
    - Defend: gain block
    - Buff: self buffs (strength, block)
    - Debuff: apply debuffs to player (weak, vulnerable, frail, etc.)
    - Heal: restore enemy HP

    The intents are set during _set_enemy_intents() from the move tables,
    which store the full intent dict including all buff/debuff effects.
    """
    for i, enemy in enumerate(state.enemies):
        if not enemy.is_alive:
            continue

        # Resolve the primary intent action
        if enemy.intent_type == "Attack" and enemy.intent_damage is not None:
            _enemy_attacks_player(state, enemy)
        elif enemy.intent_type == "Defend" and enemy.intent_block is not None:
            enemy.block += enemy.intent_block

        # Apply any secondary effects from the intent
        # These are extracted from the move table and stored in intent_* fields
        # The _set_enemy_intents() function in simulator.py stores the full intent dict
        # on the AI as _pending_intent; for combat_engine to work independently,
        # we need to store buff/debuff effects on the enemy itself.
        # For now, these are handled in _resolve_sim_intents() in simulator.py
        # but we include the logic here to be thorough.
        _apply_intent_effects(state, enemy)


def _apply_intent_effects(state: CombatState, enemy: EnemyState) -> None:
    """Apply buff/debuff effects from an enemy's intent.

    These effects are stored on the enemy by _set_enemy_intents() in simulator.py
    which extracts them from the move tables and stores them as intent_* fields
    on the EnemyState. This function applies those effects to the game state.

    Enemy self-buffs:
    - intent_self_strength: gain Strength
    - intent_self_block: gain Block (secondary effect, distinct from Defend intent)
    - intent_self_heal: restore HP (capped at max_hp)

    Player debuffs:
    - intent_player_weak: apply Weak
    - intent_player_vulnerable: apply Vulnerable
    - intent_player_frail: apply Frail
    - intent_player_constrict: apply Constrict
    - intent_player_tangled: apply Tangled
    - intent_player_shrink: reduce Strength

    All-enemy buffs:
    - intent_all_strength: all alive enemies gain Strength
    """
    # Self-buffs for enemy
    if hasattr(enemy, 'intent_self_strength') and enemy.intent_self_strength:
        enemy.powers["Strength"] = (
            enemy.powers.get("Strength", 0) + enemy.intent_self_strength
        )

    if hasattr(enemy, 'intent_self_block') and enemy.intent_self_block:
        enemy.block += enemy.intent_self_block

    if hasattr(enemy, 'intent_self_heal') and enemy.intent_self_heal:
        enemy.hp = min(enemy.hp + enemy.intent_self_heal, enemy.max_hp)

    # All-ally buffs
    if hasattr(enemy, 'intent_all_strength') and enemy.intent_all_strength:
        for e in state.enemies:
            if e.is_alive:
                e.powers["Strength"] = (
                    e.powers.get("Strength", 0) + enemy.intent_all_strength
                )

    # Player debuffs
    if hasattr(enemy, 'intent_player_weak') and enemy.intent_player_weak:
        state.player.powers["Weak"] = (
            state.player.powers.get("Weak", 0) + enemy.intent_player_weak
        )

    if hasattr(enemy, 'intent_player_vulnerable') and enemy.intent_player_vulnerable:
        state.player.powers["Vulnerable"] = (
            state.player.powers.get("Vulnerable", 0) + enemy.intent_player_vulnerable
        )

    if hasattr(enemy, 'intent_player_frail') and enemy.intent_player_frail:
        state.player.powers["Frail"] = (
            state.player.powers.get("Frail", 0) + enemy.intent_player_frail
        )

    if hasattr(enemy, 'intent_player_constrict') and enemy.intent_player_constrict:
        state.player.powers["Constrict"] = (
            state.player.powers.get("Constrict", 0) + enemy.intent_player_constrict
        )

    if hasattr(enemy, 'intent_player_tangled') and enemy.intent_player_tangled:
        state.player.powers["Tangled"] = (
            state.player.powers.get("Tangled", 0) + enemy.intent_player_tangled
        )

    if hasattr(enemy, 'intent_player_shrink') and enemy.intent_player_shrink:
        state.player.powers["Shrink"] = (
            state.player.powers.get("Shrink", 0) - enemy.intent_player_shrink
        )


def _enemy_attacks_player(state: CombatState, enemy: EnemyState) -> None:
    """Enemy attacks the player."""
    hits = enemy.intent_hits
    base_damage = enemy.intent_damage

    for _ in range(hits):
        if state.player.hp <= 0:
            break
        # Calculate damage: base + enemy Strength
        raw = base_damage + enemy.powers.get("Strength", 0)
        if raw < 0:
            raw = 0
        # Weak on enemy reduces their damage
        if enemy.powers.get("Weak", 0) > 0:
            weak_mult = 0.60 if "PAPER_KRANE" in state.relics else 0.75
            raw = math.floor(raw * weak_mult)
        # Beating Remnant: cap damage at 20 per turn
        if "BEATING_REMNANT" in state.relics:
            taken = state.player.powers.get("_damage_this_turn", 0)
            raw = min(raw, max(0, 20 - taken))
            state.player.powers["_damage_this_turn"] = taken + raw
        # Diamond Diadem: half damage if played ≤2 cards last turn
        if state.player.powers.get("_diamond_diadem_active"):
            raw = math.floor(raw * 0.5)
        # Vulnerable on player increases damage taken
        if state.player.powers.get("Vulnerable", 0) > 0:
            raw = math.floor(raw * 1.5)
        # Tank: player takes double damage
        if state.player.powers.get("Tank", 0) > 0:
            raw *= 2

        # Relic proxy: incoming damage reduction (Tungsten Rod, Diamond Diadem, etc.)
        incoming_mult = relic_effects.get_incoming_damage_reduction(state.relics)
        if incoming_mult < 1.0 and raw > 0:
            raw = max(0, math.floor(raw * incoming_mult))

        # Apply block
        if state.player.block > 0:
            if raw >= state.player.block:
                raw -= state.player.block
                state.player.block = 0
            else:
                state.player.block -= raw
                raw = 0

        # Tungsten Rod: reduce incoming damage by 1
        if "TUNGSTEN_ROD" in state.relics and raw > 0:
            raw = max(0, raw - 1)

        state.player.hp -= raw

        # Thorns on player: enemy takes damage per hit
        thorns = state.player.powers.get("Thorns", 0)
        if thorns > 0:
            enemy.hp -= thorns

        # Flame Barrier on player: enemy takes damage per hit
        flame_barrier = state.player.powers.get("Flame Barrier", 0)
        if flame_barrier > 0:
            enemy.hp -= flame_barrier


# ---------------------------------------------------------------------------
# Power ticks
# ---------------------------------------------------------------------------

def _tick_start_of_turn_powers(state: CombatState) -> None:
    """Trigger start-of-turn powers."""
    powers = state.player.powers

    # Demon Form: gain Strength
    if "Demon Form" in powers:
        powers["Strength"] = powers.get("Strength", 0) + powers["Demon Form"]

    # Ritual: gain Strength
    if "Ritual" in powers:
        powers["Strength"] = powers.get("Strength", 0) + powers["Ritual"]

    # Metallicize: gain Block (not affected by Dexterity/Frail)
    if "Metallicize" in powers:
        state.player.block += powers["Metallicize"]

    # Combust: lose HP, deal damage to all enemies
    if "Combust" in powers:
        state.player.hp -= 1
        for enemy in state.enemies:
            if enemy.is_alive:
                enemy.hp -= powers["Combust"]

    # Brutality: lose HP, draw card
    if "Brutality" in powers:
        state.player.hp -= 1
        draw_cards(state, powers["Brutality"])

    # Regen: heal and decrement
    if "Regen" in powers and powers["Regen"] > 0:
        state.player.hp = min(state.player.hp + powers["Regen"], state.player.max_hp)
        powers["Regen"] -= 1
        if powers["Regen"] <= 0:
            del powers["Regen"]

    # Noxious Fumes: apply Poison to ALL enemies
    if "Noxious Fumes" in powers:
        for enemy in state.enemies:
            if enemy.is_alive:
                enemy.powers["Poison"] = enemy.powers.get("Poison", 0) + powers["Noxious Fumes"]

    # Infinite Blades: add a Shiv to hand
    if "Infinite Blades" in powers:
        from .card_registry import _make_shiv
        for _ in range(powers["Infinite Blades"]):
            state.player.hand.append(_make_shiv())

    # Aggression: move a random Attack from discard to hand
    if "Aggression" in powers:
        attacks_in_discard = [
            c for c in state.player.discard_pile
            if c.card_type == CardType.ATTACK
        ]
        if attacks_in_discard:
            picked = attacks_in_discard[0]  # deterministic for solver
            state.player.discard_pile.remove(picked)
            state.player.hand.append(picked)

    # Tools of the Trade: draw 1 card, then discard 1 card
    if "Tools of the Trade" in powers:
        # FIXED: Added implementation of Tools of the Trade power trigger
        draw_cards(state, 1)
        if state.player.hand:
            # Heuristically discard the worst card: prefer Status/Curse > high-cost > last card
            worst_idx = 0
            worst_card = state.player.hand[0]
            for i, c in enumerate(state.player.hand):
                # Prioritize junk cards
                if c.is_junk and not worst_card.is_junk:
                    worst_idx = i
                    worst_card = c
                elif c.is_junk and worst_card.is_junk:
                    # Both junk: prefer highest cost
                    if c.cost > worst_card.cost:
                        worst_idx = i
                        worst_card = c
                elif not c.is_junk and not worst_card.is_junk:
                    # Neither junk: prefer highest cost (excess capacity)
                    if c.cost > worst_card.cost:
                        worst_idx = i
                        worst_card = c
            # Discard the worst card
            from .effects import discard_card_from_hand
            discard_card_from_hand(state, worst_idx)

    # Well-Laid Plans: Retain 1 card at end of turn
    # Implementation: simply track that retention is active (approximated as power)
    # The actual retain logic would need to be more complex in a full implementation.
    # For now, we just keep the power on the player.
    if "Well-Laid Plans" in powers:
        pass  # Engine will handle retention in discard phase if implemented

    # Automation: Every 10 cards drawn, gain 1 energy
    if "Automation" in powers:
        cards_drawn_count = state.cards_drawn_this_turn
        automation_stacks = powers.get("Automation", 0)
        # For every 10 cards drawn per stack, gain 1 energy
        if automation_stacks > 0 and cards_drawn_count >= 10:
            from .effects import gain_energy
            gain_energy(state, cards_drawn_count // 10 * automation_stacks)

    # Shadowmeld: Double block gain this turn
    # This is a turn-duration power, so just keep it on the player
    if "Shadowmeld" in powers:
        pass  # Engine will check this when calculating block gains

    # Accelerant: Poison is triggered 1 additional time
    # This is handled in tick_enemy_powers where poison ticks
    # The power is just marked; the actual multiplier logic goes in tick_enemy_powers
    if "Accelerant" in powers:
        pass  # Handled in tick_enemy_powers


def _tick_end_of_turn_powers(state: CombatState) -> None:
    """Tick down player duration-based powers at end of turn.

    Enemy debuffs and poison are ticked AFTER enemy intents resolve,
    via tick_enemy_powers(). This matches the real game order:
    player end turn → enemy acts → enemy debuffs expire → poison ticks.
    """
    # Player debuffs and turn-duration powers
    for debuff in ("Vulnerable", "Weak", "Frail", "Tangled"):
        if debuff in state.player.powers:
            state.player.powers[debuff] -= 1
            if state.player.powers[debuff] <= 0:
                del state.player.powers[debuff]

    # Shadowmeld: expires at end of turn (turn-duration)
    if "Shadowmeld" in state.player.powers:
        del state.player.powers["Shadowmeld"]


def end_combat_relics(state: CombatState) -> None:
    """Apply end-of-combat relic effects (healing, etc.). Call after combat ends.

    Dispatches through ``relic_effects.apply_end_of_combat``.
    """
    relic_effects.apply_end_of_combat(state)


def tick_enemy_powers(state: CombatState) -> None:
    """Tick enemy debuffs and poison. Call AFTER resolve_enemy_intents().

    Order matters: Weak/Vulnerable must be active during enemy attacks,
    then expire afterward. Poison deals damage after enemies act.
    """
    for enemy in state.enemies:
        if not enemy.is_alive:
            continue
        # Territorial: gain Strength equal to stacks at end of turn
        territorial = enemy.powers.get("Territorial", 0)
        if territorial > 0:
            enemy.powers["Strength"] = enemy.powers.get("Strength", 0) + territorial

        for debuff in ("Vulnerable", "Weak"):
            if debuff in enemy.powers:
                enemy.powers[debuff] -= 1
                if enemy.powers[debuff] <= 0:
                    del enemy.powers[debuff]
        # Poison: deal damage equal to stacks, then decrement by 1
        poison = enemy.powers.get("Poison", 0)
        if poison > 0:
            was_alive = enemy.is_alive
            # Accelerant: apply poison bonus (additional trigger)
            accelerant_stacks = state.player.powers.get("Accelerant", 0)
            total_poison_damage = poison * (1 + accelerant_stacks)
            enemy.hp -= total_poison_damage
            enemy.powers["Poison"] = poison - 1
            if enemy.powers["Poison"] <= 0:
                del enemy.powers["Poison"]
            if enemy.hp <= 0:
                enemy.hp = 0
                if was_alive:
                    # Death from poison: triggers with from_poison=True
                    from .effects import _on_enemy_death
                    enemy_idx = state.enemies.index(enemy)
                    _on_enemy_death(state, enemy_idx, from_poison=True)


# ---------------------------------------------------------------------------
# Combat status
# ---------------------------------------------------------------------------

def is_combat_over(state: CombatState) -> str | None:
    """Return 'win' if all enemies dead, 'lose' if player dead, None otherwise."""
    if state.player.hp <= 0:
        # Fairy in a Bottle: auto-revive at 30% max HP
        if state.player.powers.get("Fairy", 0) > 0:
            state.player.hp = max(1, int(state.player.max_hp * 0.3))
            del state.player.powers["Fairy"]
            return None
        return "lose"
    if all(not e.is_alive for e in state.enemies):
        return "win"
    return None
