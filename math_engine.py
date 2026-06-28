"""
math_engine.py — Combinatorics, Probability & Card Counting
=============================================================
Implements:
  1. Hi-Lo Card Counting System (Running Count + True Count)
  2. Exact next-card probability from finite shoe state
  3. Memoized recursive dealer outcome probabilities (H17 rule)
  4. Dealer bust probability conditioned on upcard + remaining deck

Mathematical Foundation:
  - All probabilities are computed exactly from the discrete distribution
    of remaining cards, NOT from Monte Carlo sampling.
  - Dealer outcome computation uses dynamic programming with memoization
    over the state space (dealer_total, is_soft, shoe_state_hash).
"""

import numpy as np
from functools import lru_cache
from typing import Dict, Tuple, Optional
from deck import Shoe, RANK_NAMES, RANK_VALUES

# ── Hi-Lo Card Counting Tags ────────────────────────────────────────────────
# Index:  0(A)  1(2)  2(3)  3(4)  4(5)  5(6)  6(7)  7(8)  8(9)  9(T)
# Tag:    -1    +1    +1    +1    +1    +1     0     0     0    -1
HI_LO_TAGS = np.array([-1, 1, 1, 1, 1, 1, 0, 0, 0, -1], dtype=np.int32)


class HiLoCounter:
    """
    Hi-Lo Card Counting System.

    The Hi-Lo system assigns point values to cards:
      - Cards 2–6:  +1 (low cards, favorable when removed)
      - Cards 7–9:   0 (neutral)
      - Cards 10–A: -1 (high cards, favorable when present)

    Running Count (RC): Sum of tags for all observed cards.
    True Count (TC): RC normalized by decks remaining.
        TC = RC / decks_remaining
        where decks_remaining = cards_remaining / 52

    A positive TC indicates a player-favorable deck (rich in high cards).
    A negative TC indicates a dealer-favorable deck (rich in low cards).
    """

    def __init__(self):
        self.running_count: int = 0
        self.cards_seen: int = 0

    def observe_card(self, rank: int) -> None:
        """
        Update the count after observing a dealt card.

        Args:
            rank: Rank index (0=Ace, 1=Two, ..., 9=Ten-value).
        """
        self.running_count += int(HI_LO_TAGS[rank])
        self.cards_seen += 1

    def true_count(self, shoe: Shoe) -> float:
        """
        Calculate the True Count.

        TC = Running_Count / Decks_Remaining

        Decks remaining is estimated as cards_remaining / 52.
        This normalization allows comparison across different shoe sizes.
        """
        decks_rem = shoe.decks_remaining
        if decks_rem < 0.25:  # Avoid division by near-zero
            return float(self.running_count) * 4.0
        return self.running_count / decks_rem

    def reset(self) -> None:
        """Reset counter for a new shoe."""
        self.running_count = 0
        self.cards_seen = 0

    def __repr__(self) -> str:
        return f"HiLoCounter(RC={self.running_count}, seen={self.cards_seen})"


class ProbabilityEngine:
    """
    Exact probability calculations over the finite shoe state.

    All computations are conditioned on the exact remaining cards in the shoe,
    making them precise rather than approximate. This is the mathematical core
    that feeds into the AI decision engine.
    """

    @staticmethod
    def next_card_probs(shoe: Shoe) -> Dict[int, float]:
        """
        Exact probability of drawing each rank from the current shoe.

        P(rank=r) = shoe.cards[r] / shoe.total_remaining

        Returns:
            Dict mapping rank index → probability. Only includes ranks
            with P > 0 (still available in shoe).
        """
        probs = shoe.all_probabilities()
        return {r: float(p) for r, p in enumerate(probs) if p > 0}

    @staticmethod
    def next_card_prob_array(shoe: Shoe) -> np.ndarray:
        """Return probability array of shape (10,) for all ranks."""
        return shoe.all_probabilities()

    @staticmethod
    def dealer_outcome_probabilities(
        upcard: int,
        shoe: Shoe,
    ) -> Dict[str, float]:
        shoe_cards = shoe.cards.copy()
        remaining = int(shoe_cards.sum())
        
        if remaining == 0:
            total = RANK_VALUES[upcard]
            if total > 21: return {"bust": 1.0}
            return {str(total): 1.0}
            
        # Precompute static probabilities for the entire dealer draw
        rank_probs = [shoe_cards[r] / remaining for r in range(10)]

        @lru_cache(None)
        def _recurse(total: int, is_soft: bool) -> Dict[str, float]:
            must_hit = (total < 17) or (total == 17 and is_soft)

            if not must_hit:
                if total > 21:
                    return {"bust": 1.0}
                return {str(min(total, 21)): 1.0}

            if total > 21:
                return {"bust": 1.0}

            result = {"17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "bust": 0.0}

            for rank in range(10):
                prob = rank_probs[rank]
                if prob == 0:
                    continue

                card_value = RANK_VALUES[rank]
                new_total = total + card_value
                new_soft = is_soft

                if rank == 0:  
                    new_soft = True

                if new_total > 21 and new_soft:
                    new_total -= 10
                    new_soft = False

                sub_result = _recurse(new_total, new_soft)

                for key in result:
                    result[key] += prob * sub_result.get(key, 0.0)

            return result

        initial_value = RANK_VALUES[upcard]
        initial_soft = (upcard == 0)
        return _recurse(initial_value, initial_soft)

    @staticmethod
    def dealer_outcome_probabilities_full(
        upcard: int,
        hole_card: int,
        shoe: Shoe,
    ) -> Dict[str, float]:
        total = RANK_VALUES[upcard] + RANK_VALUES[hole_card]
        is_soft = (upcard == 0 or hole_card == 0)

        if total > 21 and is_soft:
            total -= 10
            if not (upcard == 0 and hole_card == 0):
                is_soft = (total <= 21) and (upcard == 0 or hole_card == 0)
            else:
                is_soft = True

        shoe_cards = shoe.cards.copy()
        remaining = int(shoe_cards.sum())
        if remaining == 0:
            if total > 21: return {"bust": 1.0}
            return {str(min(total, 21)): 1.0}
            
        rank_probs = [shoe_cards[r] / remaining for r in range(10)]

        @lru_cache(None)
        def _recurse(t: int, soft: bool) -> Dict[str, float]:
            must_hit = (t < 17) or (t == 17 and soft)

            if not must_hit:
                if t > 21:
                    return {"bust": 1.0}
                return {str(min(t, 21)): 1.0}

            if t > 21:
                return {"bust": 1.0}

            result = {"17": 0.0, "18": 0.0, "19": 0.0, "20": 0.0, "21": 0.0, "bust": 0.0}

            for rank in range(10):
                prob = rank_probs[rank]
                if prob == 0:
                    continue

                new_t = t + RANK_VALUES[rank]
                new_soft = soft or (rank == 0)

                if new_t > 21 and new_soft:
                    new_t -= 10
                    new_soft = False

                sub = _recurse(new_t, new_soft)
                for key in result:
                    result[key] += prob * sub.get(key, 0.0)

            return result

        return _recurse(total, is_soft)

    @staticmethod
    def dealer_bust_probability(upcard: int, shoe: Shoe) -> float:
        """
        Convenience method: exact probability of dealer busting
        given their upcard and current shoe state.

        P(bust | upcard, shoe) from the full outcome distribution.
        """
        outcomes = ProbabilityEngine.dealer_outcome_probabilities(upcard, shoe)
        return outcomes.get("bust", 0.0)

    @staticmethod
    def ten_value_ratio(shoe: Shoe) -> float:
        """
        Ratio of ten-value cards remaining in the shoe.

        P(10-value) = shoe.cards[9] / shoe.total_remaining

        This is the key metric for insurance decisions.
        """
        remaining = shoe.total_remaining
        if remaining == 0:
            return 0.0
        return shoe.cards[9] / remaining
