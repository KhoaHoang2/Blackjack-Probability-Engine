"""
deck.py — Array-Based Shoe Management for Blackjack
=====================================================
Implements a 6-deck shoe using a NumPy array of shape (10,) where each
index represents a card rank:
    Index:  0   1   2   3   4   5   6   7   8   9
    Rank:   A   2   3   4   5   6   7   8   9  T(10/J/Q/K)

A fresh 6-deck shoe has 24 of each rank except ten-values (96 total).
Cards are dealt by weighted random sampling from remaining counts.
"""

import numpy as np
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
NUM_DECKS = 6
CARDS_PER_DECK = 52
TOTAL_CARDS = NUM_DECKS * CARDS_PER_DECK  # 312

# Rank names for display
RANK_NAMES = ["A", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
RANK_VALUES = [11, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Ace=11 initially (soft)

# Fresh 6-deck distribution: 24 each for A-9, 96 for ten-values (10/J/Q/K)
FRESH_SHOE = np.array([24, 24, 24, 24, 24, 24, 24, 24, 24, 96], dtype=np.int32)

# Default penetration: reshuffle when 75% of shoe is dealt
DEFAULT_PENETRATION = 0.75


class Shoe:
    """
    Array-based finite shoe for Blackjack.

    Tracks exact remaining card counts as a 10-element integer array.
    Deals cards via weighted random sampling (not infinite replacement).

    Attributes:
        cards (np.ndarray): Shape (10,) — count of each rank remaining.
        num_decks (int): Number of decks in the shoe.
        penetration_threshold (float): Fraction dealt before reshuffle signal.
        rng (np.random.Generator): Random number generator for reproducibility.
    """

    def __init__(
        self,
        num_decks: int = NUM_DECKS,
        penetration_threshold: float = DEFAULT_PENETRATION,
        seed: Optional[int] = None,
    ):
        self.num_decks = num_decks
        self.penetration_threshold = penetration_threshold
        self.rng = np.random.default_rng(seed)

        # Build fresh shoe scaled by number of decks
        base = np.array([4, 4, 4, 4, 4, 4, 4, 4, 4, 16], dtype=np.int32)
        self._fresh = base * num_decks
        self.cards = self._fresh.copy()

    @property
    def total_cards(self) -> int:
        """Total cards originally in the shoe."""
        return self.num_decks * CARDS_PER_DECK

    @property
    def total_remaining(self) -> int:
        """Number of cards still in the shoe."""
        return int(self.cards.sum())

    @property
    def cards_dealt(self) -> int:
        """Number of cards dealt so far."""
        return self.total_cards - self.total_remaining

    @property
    def decks_remaining(self) -> float:
        """Estimated decks remaining (used for True Count calculation)."""
        return self.total_remaining / CARDS_PER_DECK

    @property
    def penetration(self) -> float:
        """Fraction of the shoe that has been dealt."""
        return self.cards_dealt / self.total_cards

    @property
    def needs_reshuffle(self) -> bool:
        """True if penetration exceeds threshold — time to reshuffle."""
        return self.penetration >= self.penetration_threshold

    def deal_card(self) -> int:
        """
        Deal one card from the shoe via weighted random sampling.

        Returns:
            int: Rank index (0=Ace, 1=Two, ..., 9=Ten-value).

        Raises:
            RuntimeError: If shoe is empty.
        """
        remaining = self.total_remaining
        if remaining == 0:
            raise RuntimeError("Shoe is empty — cannot deal.")

        # Weighted random sample: probability proportional to remaining counts
        probabilities = self.cards.astype(np.float64) / remaining
        rank = self.rng.choice(10, p=probabilities)
        self.cards[rank] -= 1
        return int(rank)

    def deal_specific(self, rank: int) -> int:
        """
        Deal a specific rank from the shoe (used for testing / rigged deals).

        Args:
            rank: Rank index (0-9) to deal.

        Returns:
            int: The rank dealt.

        Raises:
            ValueError: If that rank is exhausted.
        """
        if self.cards[rank] <= 0:
            raise ValueError(f"No {RANK_NAMES[rank]}s remaining in shoe.")
        self.cards[rank] -= 1
        return rank

    def return_card(self, rank: int) -> None:
        """Return a card to the shoe (undo a deal)."""
        self.cards[rank] += 1

    def peek_rank_available(self, rank: int) -> bool:
        """Check if at least one card of this rank is available."""
        return self.cards[rank] > 0

    def rank_probability(self, rank: int) -> float:
        """
        Exact probability of drawing a specific rank.

        P(rank) = cards[rank] / total_remaining
        """
        remaining = self.total_remaining
        if remaining == 0:
            return 0.0
        return self.cards[rank] / remaining

    def all_probabilities(self) -> np.ndarray:
        """
        Return probability distribution over all 10 ranks.

        Returns:
            np.ndarray: Shape (10,) — P(rank_i) for i in [0..9].
        """
        remaining = self.total_remaining
        if remaining == 0:
            return np.zeros(10, dtype=np.float64)
        return self.cards.astype(np.float64) / remaining

    def reset(self) -> None:
        """Reinitialize to a full fresh shoe."""
        self.cards = self._fresh.copy()

    def copy(self) -> "Shoe":
        """Create a deep copy of the shoe (for hypothetical calculations)."""
        new_shoe = Shoe.__new__(Shoe)
        new_shoe.num_decks = self.num_decks
        new_shoe.penetration_threshold = self.penetration_threshold
        new_shoe.rng = self.rng
        new_shoe._fresh = self._fresh.copy()
        new_shoe.cards = self.cards.copy()
        return new_shoe

    def state_tuple(self) -> tuple:
        """
        Hashable state representation for memoization.

        Returns:
            tuple: Immutable tuple of card counts.
        """
        return tuple(self.cards.tolist())

    def __repr__(self) -> str:
        counts = ", ".join(
            f"{RANK_NAMES[i]}:{self.cards[i]}" for i in range(10)
        )
        return (
            f"Shoe({self.num_decks}deck | "
            f"{self.total_remaining}/{self.total_cards} remaining | "
            f"pen={self.penetration:.1%} | {counts})"
        )
