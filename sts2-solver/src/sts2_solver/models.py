from __future__ import annotations

from dataclasses import dataclass, field

from .constants import CardType, TargetType


# Quest / loot carry cards. Classified as Status by the game but they
# sit inert in the deck with no combat effect. NOT junk (we never want
# them *removed* from the deck), but in discard-from-hand prompts they
# are the best target after real junk — discarding them loses nothing.
CARRY_CARGO_STATUS_IDS = frozenset({
    "SPOILS_MAP",
    "LANTERN_KEY",
    "BYRDONIS_EGG",
})


@dataclass
class Card:
    """Immutable card template. Loaded from JSON at startup."""

    id: str
    name: str
    cost: int
    card_type: CardType
    target: TargetType
    upgraded: bool = False

    # Effect fields (None means card doesn't have this effect)
    damage: int | None = None
    block: int | None = None
    hit_count: int = 1
    powers_applied: tuple[tuple[str, int], ...] = ()
    cards_draw: int = 0
    energy_gain: int = 0
    hp_loss: int = 0

    # Keywords and tags
    keywords: frozenset[str] = field(default_factory=frozenset)
    tags: frozenset[str] = field(default_factory=frozenset)

    # Cards this card spawns/adds
    spawns_cards: tuple[str, ...] = ()

    # X-cost
    is_x_cost: bool = False

    # Card vars — raw effect parameters from JSON (e.g. PoisonPerTurn, Accelerant)
    vars: dict[str, int | float] = field(default_factory=dict)

    # Rarity tier (Common, Uncommon, Rare, Ancient)
    rarity: str = "Common"

    # Card description text (for fallback effect detection)
    description: str = ""

    @property
    def exhausts(self) -> bool:
        return "Exhaust" in self.keywords

    @property
    def innate(self) -> bool:
        return "Innate" in self.keywords

    @property
    def ethereal(self) -> bool:
        return "Ethereal" in self.keywords

    @property
    def retain(self) -> bool:
        return "Retain" in self.keywords

    @property
    def is_junk(self) -> bool:
        """Is this card worthless to keep in hand (should be discarded first)?

        Rules:
          - All Curses are junk (curses always punish you).
          - Statuses are junk UNLESS they're an allowlisted "carry-cargo"
            card that encodes quest / loot info (Spoils Map, Lantern Key,
            Byrdonis Egg). Those are handled by is_carry_cargo — see the
            note there for why they're a separate tier from real junk.
          - All Attack / Skill / Power cards are NOT junk.

        This covers every STS2 status/curse with a single small allowlist
        instead of a sprawling known-junk blacklist, and has no false
        positives on real player cards (Attack/Skill/Power never match).
        """
        if self.card_type == CardType.CURSE:
            return True
        if self.card_type != CardType.STATUS:
            return False
        return self.id not in CARRY_CARGO_STATUS_IDS

    @property
    def is_carry_cargo(self) -> bool:
        """Is this an inert quest-status card (Spoils Map / Lantern Key /
        Byrdonis Egg)?

        These sit in the deck with no combat effect but carry quest or
        loot info that shouldn't be removed from the deck. They occupy
        a middle tier for discard-from-hand decisions:

            real junk  (Wound, Clumsy, ...)   ← discard first
            carry cargo (Spoils Map, ...)     ← discard next
            real cards  (Strike, Sly, ...)    ← keep

        Discarding a carry-cargo card from combat-hand is free — it goes
        to the discard pile and comes back to the draw pile next shuffle,
        losing nothing — while discarding a Strike or a Sly card loses
        real DPS. So when there's no real junk in hand, carry cargo
        should still be the preferred discard target over any playable
        card.
        """
        return self.id in CARRY_CARGO_STATUS_IDS


@dataclass
class PlayerState:
    hp: int
    max_hp: int
    block: int = 0
    energy: int = 3
    max_energy: int = 3
    powers: dict[str, int] = field(default_factory=dict)

    hand: list[Card] = field(default_factory=list)
    draw_pile: list[Card] = field(default_factory=list)
    discard_pile: list[Card] = field(default_factory=list)
    exhaust_pile: list[Card] = field(default_factory=list)
    potions: list[dict] = field(default_factory=list)  # [{"name": ..., "heal": 20}, ...]

    burst_count: int = 0  # Number of Skills to play twice (Burst card)
    no_draw_this_turn: bool = False  # Bullet Time: prevent drawing cards
    all_cards_free: bool = False  # Bullet Time: make all cards cost 0


@dataclass
class EnemyState:
    id: str
    name: str
    hp: int
    max_hp: int
    block: int = 0
    powers: dict[str, int] = field(default_factory=dict)

    # Current intent (known from game state)
    intent_type: str | None = None  # "Attack", "Defend", "Buff", "Debuff", "StatusCard"
    intent_damage: int | None = None
    intent_hits: int = 1
    intent_block: int | None = None

    # Buff/debuff effects from the move table
    # These are populated by _set_enemy_intents() in simulator.py
    intent_self_strength: int | None = None  # Strength to gain
    intent_self_block: int | None = None  # Block to gain (secondary effect, distinct from Defend intent)
    intent_self_heal: int | None = None  # HP to restore
    intent_all_strength: int | None = None  # Strength for all allies
    intent_player_weak: int | None = None  # Weak to apply to player
    intent_player_vulnerable: int | None = None  # Vulnerable to apply to player
    intent_player_frail: int | None = None  # Frail to apply to player
    intent_player_constrict: int | None = None  # Constrict to apply to player
    intent_player_tangled: int | None = None  # Tangled to apply to player
    intent_player_shrink: int | None = None  # Shrink to apply to player

    # Predicted future intents (from move table lookahead)
    predicted_intents: list[dict] = field(default_factory=list)

    @property
    def is_alive(self) -> bool:
        return self.hp > 0


@dataclass
class PendingChoice:
    """A sub-decision the player must make before combat can proceed.

    Created by card effects that require player input (e.g., "discard 1 card").
    When set on CombatState, enumerate_actions() returns only choose_card actions.
    """

    choice_type: str        # "discard_from_hand", "choose_from_discard", "choose_from_hand"
    num_choices: int         # How many cards must be chosen (1 for Survivor, 2 for Hidden Daggers)
    source_card_id: str      # Card that triggered this choice (for post-resolve hooks)
    valid_indices: list[int] | None = None  # Restrict which indices are valid (None = all)
    chosen_so_far: list[int] = field(default_factory=list)  # For multi-select


@dataclass
class CombatState:
    player: PlayerState
    enemies: list[EnemyState]
    turn: int = 0
    cards_played_this_turn: int = 0
    attacks_played_this_turn: int = 0
    cards_drawn_this_turn: int = 0  # Total draw effects triggered (for scoring)
    discards_this_turn: int = 0  # Cards discarded this turn (for Memento Mori, etc.)
    last_x_cost: int = 0  # Energy spent on the most recent X-cost card
    relics: frozenset[str] = field(default_factory=frozenset)  # Relic IDs held
    floor: int = 0  # Current floor number (for scaling bonuses)
    gold: int = 0  # Current gold (used by non-combat decision heads)
    pending_choice: PendingChoice | None = None  # Sub-choice awaiting player input
