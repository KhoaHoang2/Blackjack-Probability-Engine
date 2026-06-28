"""
ai_advisor.py — AI Decision Engine & EV Calculator
=====================================================
Implements:
  1. Expected Value (EV) calculation for all player actions:
     Hit, Stand, Double, Split, Surrender, Insurance
  2. Kelly Criterion bet sizing based on True Count
  3. Optimal move recommendation (highest EV)

Mathematical Foundation:
  All EV calculations are derived from exact discrete probabilities
  over the remaining shoe composition. No lookup tables or pre-computed
  basic strategy charts are used.

  EV(action) = Σ P(outcome) × payout(outcome)

  The player is assumed to play optimally after each decision point
  (i.e., the EV of hitting incorporates the fact that the player will
  subsequently make the best available choice at each follow-up).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from deck import Shoe, RANK_VALUES, RANK_NAMES
from engine import Hand, Action, BlackjackGame
from math_engine import ProbabilityEngine, HiLoCounter


class EVCalculator:
    """
    Calculates Expected Value for all available Blackjack actions.

    Uses recursive exact probability computation over the finite shoe
    for single-hand interactive play. Results are normalized to the
    original bet size (EV of 1.0 = win exactly your bet).

    The recursive depth is bounded by the maximum cards a player/dealer
    can draw (at most ~10 cards before busting), making exact computation
    tractable for interactive use.
    """

    # Cache for dealer outcome distributions to avoid recomputation
    _dealer_cache: Dict[Tuple, Dict[str, float]] = {}

    @classmethod
    def clear_cache(cls):
        """Clear the dealer outcome cache."""
        cls._dealer_cache.clear()

    @staticmethod
    def get_dealer_outcomes(
        upcard: int, shoe: Shoe
    ) -> Dict[str, float]:
        """
        Get dealer outcome distribution, with caching.

        Cache key is (upcard, shoe_state_tuple) for O(1) lookup on
        repeated queries with the same shoe state.
        """
        cache_key = (upcard, shoe.state_tuple())
        if cache_key in EVCalculator._dealer_cache:
            return EVCalculator._dealer_cache[cache_key]

        outcomes = ProbabilityEngine.dealer_outcome_probabilities(upcard, shoe)
        EVCalculator._dealer_cache[cache_key] = outcomes
        return outcomes

    @staticmethod
    def ev_stand(
        player_total: int,
        dealer_outcomes: Dict[str, float],
    ) -> float:
        """
        EV of standing on the current total.

        EV_stand = P(dealer_bust) × (+1)
                 + Σ_{d=17}^{21} P(dealer=d) × {
                     +1  if player > d
                      0  if player == d
                     -1  if player < d
                   }

        All values normalized to 1 unit bet.

        Args:
            player_total: Player's current hand total.
            dealer_outcomes: Probability distribution of dealer final totals.

        Returns:
            float: Expected value in units of the bet.
        """
        ev = 0.0

        # Dealer bust → player wins
        ev += dealer_outcomes.get("bust", 0.0) * 1.0

        # Compare against each possible dealer total
        for d_total_str in ["17", "18", "19", "20", "21"]:
            d_total = int(d_total_str)
            prob = dealer_outcomes.get(d_total_str, 0.0)

            if player_total > d_total:
                ev += prob * 1.0   # Win
            elif player_total == d_total:
                ev += prob * 0.0   # Push
            else:
                ev += prob * (-1.0)  # Lose

        return ev

    @staticmethod
    def ev_hit(
        player_cards: List[int],
        dealer_upcard: int,
        shoe: Shoe,
        dealer_outcomes: Optional[Dict[str, float]] = None,
        depth: int = 0,
        max_depth: int = 15,
    ) -> float:
        """
        EV of hitting (recursive, with optimal subsequent play).

        EV_hit = Σ_{rank r} P(r | shoe) × {
            if new_total > 21:  -1  (bust)
            else: max(EV_stand(new), EV_hit(new))
        }

        The player plays optimally after hitting: at each subsequent
        decision point, they choose the action with highest EV
        (stand or hit again). This models perfect play recursion.

        Args:
            player_cards: Current player card list (rank indices).
            dealer_upcard: Dealer's visible card rank index.
            shoe: Current shoe state.
            dealer_outcomes: Pre-computed dealer outcome distribution.
            depth: Current recursion depth (for safety cutoff).
            max_depth: Maximum recursion depth.

        Returns:
            float: Expected value of hitting, in bet units.
        """
        if depth > max_depth:
            # Safety cutoff — approximate by standing
            total = _hand_total(player_cards)
            if total > 21:
                return -1.0
            if dealer_outcomes is None:
                dealer_outcomes = EVCalculator.get_dealer_outcomes(dealer_upcard, shoe)
            return EVCalculator.ev_stand(total, dealer_outcomes)

        remaining = shoe.total_remaining
        if remaining == 0:
            return 0.0

        if dealer_outcomes is None:
            dealer_outcomes = EVCalculator.get_dealer_outcomes(dealer_upcard, shoe)

        ev = 0.0
        probs = shoe.all_probabilities()

        for rank in range(10):
            if probs[rank] == 0:
                continue

            prob = probs[rank]

            # Simulate drawing this card
            new_cards = player_cards + [rank]
            new_total = _hand_total(new_cards)

            if new_total > 21:
                # Bust
                ev += prob * (-1.0)
            elif new_total == 21:
                # 21 — must stand, optimal
                shoe.cards[rank] -= 1
                new_dealer = EVCalculator.get_dealer_outcomes(dealer_upcard, shoe)
                stand_ev = EVCalculator.ev_stand(21, new_dealer)
                shoe.cards[rank] += 1
                ev += prob * stand_ev
            else:
                # Can hit or stand — take the best
                shoe.cards[rank] -= 1
                new_dealer = EVCalculator.get_dealer_outcomes(dealer_upcard, shoe)
                stand_ev = EVCalculator.ev_stand(new_total, new_dealer)
                hit_ev = EVCalculator.ev_hit(
                    new_cards, dealer_upcard, shoe, new_dealer, depth + 1, max_depth
                )
                shoe.cards[rank] += 1

                ev += prob * max(stand_ev, hit_ev)

        return ev

    @staticmethod
    def ev_double(
        player_cards: List[int],
        dealer_upcard: int,
        shoe: Shoe,
    ) -> float:
        """
        EV of doubling down.

        Double: bet is doubled, exactly one card drawn, then forced to stand.

        EV_double = 2 × Σ_{rank r} P(r | shoe) × EV_stand(new_total, dealer_outcomes)

        The factor of 2 reflects the doubled bet. We normalize to the
        original bet, so doubling with +EV means the expected gain is
        relative to the original wager.

        Args:
            player_cards: Current player cards.
            dealer_upcard: Dealer's upcard rank index.
            shoe: Current shoe state.

        Returns:
            float: Expected value in original bet units.
        """
        remaining = shoe.total_remaining
        if remaining == 0:
            return 0.0

        ev = 0.0
        probs = shoe.all_probabilities()

        for rank in range(10):
            if probs[rank] == 0:
                continue

            prob = probs[rank]
            new_cards = player_cards + [rank]
            new_total = _hand_total(new_cards)

            if new_total > 21:
                # Bust on double — lose 2× bet
                ev += prob * (-2.0)
            else:
                shoe.cards[rank] -= 1
                dealer_outcomes = EVCalculator.get_dealer_outcomes(dealer_upcard, shoe)
                stand_ev = EVCalculator.ev_stand(new_total, dealer_outcomes)
                shoe.cards[rank] += 1
                # Multiply by 2 for the doubled bet
                ev += prob * (2.0 * stand_ev)

        return ev

    @staticmethod
    def ev_split(
        pair_rank: int,
        dealer_upcard: int,
        shoe: Shoe,
        num_current_hands: int = 1,
        is_aces: bool = False,
    ) -> float:
        """
        EV of splitting a pair.

        Split creates two independent hands, each starting with one card
        of the pair + one new card drawn from the shoe.

        EV_split = 2 × EV(single_split_hand)

        For split Aces: each hand gets exactly one card, then must stand.
        For other pairs: full play (hit/stand/double) available, and
        re-splitting is possible up to 4 total hands.

        Simplification: We approximate by computing the EV of a single
        post-split hand and doubling it. The two hands are treated as
        independent (which is a standard approximation since their
        correlation through the shared shoe has minimal impact).

        Args:
            pair_rank: Rank index of the pair card.
            dealer_upcard: Dealer's upcard rank index.
            shoe: Current shoe state.
            num_current_hands: Current number of hands (for re-split limit).
            is_aces: Whether splitting aces.

        Returns:
            float: Expected value in original bet units (for the total
                   cost of both split hands, i.e., 2× original bet).
        """
        remaining = shoe.total_remaining
        if remaining == 0:
            return 0.0

        # EV of a single split hand
        single_hand_ev = 0.0
        probs = shoe.all_probabilities()

        for rank in range(10):
            if probs[rank] == 0:
                continue

            prob = probs[rank]
            new_cards = [pair_rank, rank]
            new_total = _hand_total(new_cards)

            shoe.cards[rank] -= 1

            if is_aces:
                # Split aces: one card only, must stand
                # 10+A=21 (not blackjack) — just a regular 21
                if new_total > 21:
                    hand_ev = -1.0
                else:
                    dealer_outcomes = EVCalculator.get_dealer_outcomes(dealer_upcard, shoe)
                    hand_ev = EVCalculator.ev_stand(new_total, dealer_outcomes)
            else:
                # Full play available
                dealer_outcomes = EVCalculator.get_dealer_outcomes(dealer_upcard, shoe)

                if new_total > 21:
                    hand_ev = -1.0
                elif new_total == 21:
                    hand_ev = EVCalculator.ev_stand(21, dealer_outcomes)
                else:
                    # Best of stand, hit, double
                    stand_ev = EVCalculator.ev_stand(new_total, dealer_outcomes)
                    hit_ev = EVCalculator.ev_hit(
                        new_cards, dealer_upcard, shoe, dealer_outcomes
                    )
                    double_ev = EVCalculator.ev_double(
                        new_cards, dealer_upcard, shoe
                    )
                    hand_ev = max(stand_ev, hit_ev, double_ev)

                    # Check for re-split possibility
                    if (rank == pair_rank
                            and num_current_hands + 1 < 4
                            and not is_aces):
                        resplit_ev = EVCalculator.ev_split(
                            pair_rank, dealer_upcard, shoe,
                            num_current_hands + 1, False
                        )
                        hand_ev = max(hand_ev, resplit_ev)

            shoe.cards[rank] += 1
            single_hand_ev += prob * hand_ev

        # Total EV = 2 hands × single hand EV (cost is 2× original bet)
        return 2.0 * single_hand_ev

    @staticmethod
    def ev_surrender() -> float:
        """
        EV of Late Surrender.

        Surrender returns half the bet. Net loss = -0.5 bet.

        EV_surrender = -0.5

        This is always -0.5 regardless of shoe state, making it a
        useful reference point: surrender when all other EVs < -0.5.
        """
        return -0.5

    @staticmethod
    def ev_insurance(shoe: Shoe) -> float:
        """
        EV of the insurance side-bet.

        Insurance costs ½ bet and pays 2:1 if dealer has blackjack.
        Dealer has BJ when hole card is a 10-value.

        P(10) = shoe.cards[9] / shoe.total_remaining
        EV_insurance = P(10) × (+1.0) + (1-P(10)) × (-0.5)

        Derivation:
          - Cost: 0.5 bet
          - If dealer has BJ (hole=10): pays 2:1 on the 0.5 bet = +1.0
            Net from insurance alone: +1.0 - 0.5 = +0.5? No:
            Insurance pays 2:1 on the insurance bet.
            Win = 2 × 0.5 = 1.0. Net = 1.0 - 0.5 = +0.5? 
            
            Actually, let's be precise:
            - You place insurance bet of 0.5
            - If dealer BJ: you get 0.5 back + 2×0.5 = 1.0 payout = net +1.0
            - If no dealer BJ: you lose 0.5 = net -0.5
            
            EV = P(10) × 1.0 + (1-P(10)) × (-0.5)

        Insurance is favorable when P(10) > 1/3.

        Args:
            shoe: Current shoe (hole card not yet dealt/revealed).

        Returns:
            float: EV of insurance in units of half the original bet.
        """
        remaining = shoe.total_remaining
        if remaining == 0:
            return 0.0

        p_ten = shoe.cards[9] / remaining
        return p_ten * 1.0 + (1.0 - p_ten) * (-0.5)

    @staticmethod
    def all_evs(
        game: BlackjackGame,
        shoe: Shoe,
    ) -> Dict[str, float]:
        """
        Calculate EV for all currently available actions.

        Returns a dict mapping action names to their expected values.
        The optimal action is the one with highest EV.

        Args:
            game: Current game state.
            shoe: Current shoe state.

        Returns:
            Dict[str, float]: {action_name: ev_value}
        """
        actions = game.available_actions()
        hand = game.active_hand

        if hand is None or not actions:
            return {}

        results = {}
        player_cards = hand.cards.copy()
        upcard = game.dealer_upcard

        if upcard is None:
            return {}

        # Pre-compute dealer outcomes once
        dealer_outcomes = EVCalculator.get_dealer_outcomes(upcard, shoe)
        player_total = hand.total

        for action in actions:
            if action == Action.STAND:
                results["STAND"] = EVCalculator.ev_stand(player_total, dealer_outcomes)

            elif action == Action.HIT:
                results["HIT"] = EVCalculator.ev_hit(
                    player_cards, upcard, shoe, dealer_outcomes
                )

            elif action == Action.DOUBLE:
                results["DOUBLE"] = EVCalculator.ev_double(
                    player_cards, upcard, shoe
                )

            elif action == Action.SPLIT:
                pair_rank = player_cards[0]
                results["SPLIT"] = EVCalculator.ev_split(
                    pair_rank, upcard, shoe,
                    num_current_hands=len(game.player_hands),
                    is_aces=(pair_rank == 0),
                )

            elif action == Action.SURRENDER:
                results["SURRENDER"] = EVCalculator.ev_surrender()

            elif action == Action.INSURANCE:
                results["INSURANCE"] = EVCalculator.ev_insurance(shoe)

        return results

    @staticmethod
    def optimal_action(evs: Dict[str, float]) -> Optional[str]:
        """Return the action name with the highest EV."""
        if not evs:
            return None
        return max(evs, key=evs.get)


class KellyCriterion:
    """
    Kelly Criterion bet sizing for Blackjack card counting.

    The Kelly fraction determines the optimal bet as a fraction of
    bankroll to maximize long-run growth rate (geometric mean return).

    Kelly Fraction:
        f* = edge / odds

    For Blackjack with even-money payoff:
        f* = edge / 1.0 = edge

    Player Edge Estimation:
        Base house edge for 6-deck H17 with full rules ≈ -0.46%
        Each unit of True Count shifts edge by ≈ +0.5%

        edge(TC) = -0.0046 + 0.005 × TC

    This is the standard approximation from card counting theory
    (Stanford Wong, "Professional Blackjack"). The exact values
    depend on the specific rule set, but this is a well-calibrated
    estimate for the H17/DAS/LS rules we implement.

    Optimal Bet:
        bet = bankroll × f* × kelly_divisor
        clamped to [min_bet, max_bet]

    We use a fractional Kelly (divisor > 1) to reduce variance
    at the cost of slightly lower long-run growth.
    """

    # Base house edge for 6-deck, H17, DAS, LS, re-split to 4
    BASE_EDGE = -0.0046

    # Edge shift per unit of True Count
    EDGE_PER_TC = 0.005

    # Fractional Kelly divisor (2.0 = half-Kelly, common in practice)
    KELLY_DIVISOR = 2.0

    @staticmethod
    def estimate_edge(true_count: float) -> float:
        """
        Estimate player edge from True Count.

        edge = BASE_EDGE + EDGE_PER_TC × TC
             = -0.46% + 0.5% × TC

        Positive edge → player advantage → bet more.
        Negative edge → house advantage → bet minimum.
        """
        return KellyCriterion.BASE_EDGE + KellyCriterion.EDGE_PER_TC * true_count

    @staticmethod
    def optimal_bet(
        true_count: float,
        bankroll: float,
        min_bet: float = 10.0,
        max_bet: float = 500.0,
    ) -> float:
        """
        Calculate optimal bet size using fractional Kelly Criterion.

        f* = max(0, edge) / KELLY_DIVISOR
        bet = bankroll × f*
        bet = clamp(bet, min_bet, max_bet)

        When edge is negative, bet the table minimum.

        Args:
            true_count: Current Hi-Lo True Count.
            bankroll: Current bankroll.
            min_bet: Table minimum bet.
            max_bet: Table maximum bet.

        Returns:
            float: Recommended bet size.
        """
        edge = KellyCriterion.estimate_edge(true_count)

        if edge <= 0:
            return min_bet  # No edge → minimum bet

        # Kelly fraction (divided for safety)
        kelly_frac = edge / KellyCriterion.KELLY_DIVISOR
        bet = bankroll * kelly_frac

        # Clamp to table limits
        bet = max(min_bet, min(bet, max_bet, bankroll))

        # Round to nearest dollar
        return round(bet, 0)

    @staticmethod
    def bet_info(
        true_count: float,
        bankroll: float,
        min_bet: float = 10.0,
        max_bet: float = 500.0,
    ) -> Dict[str, float]:
        """
        Full bet sizing information for the HUD.

        Returns:
            Dict with edge, kelly_fraction, recommended_bet.
        """
        edge = KellyCriterion.estimate_edge(true_count)
        kelly_frac = max(0, edge) / KellyCriterion.KELLY_DIVISOR
        bet = KellyCriterion.optimal_bet(true_count, bankroll, min_bet, max_bet)

        return {
            "edge_pct": edge * 100,
            "kelly_fraction": kelly_frac,
            "recommended_bet": bet,
            "bet_multiple": bet / min_bet if min_bet > 0 else 1.0,
        }


# ── Helper Functions ─────────────────────────────────────────────────────────

def _hand_total(cards: List[int]) -> int:
    """
    Calculate best hand total from a list of rank indices.

    Same logic as Hand.total but operates on a plain list for
    performance in recursive EV calculations.

    Aces (rank 0) start as 11, downgraded to 1 if bust.
    """
    total = sum(RANK_VALUES[r] for r in cards)
    aces = sum(1 for r in cards if r == 0)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return total


def _hand_is_soft(cards: List[int]) -> bool:
    """Check if a hand (card list) is soft."""
    total = sum(RANK_VALUES[r] for r in cards)
    aces = sum(1 for r in cards if r == 0)
    while total > 21 and aces > 0:
        total -= 10
        aces -= 1
    return aces > 0
