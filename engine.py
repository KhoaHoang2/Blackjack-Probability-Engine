"""
engine.py — Blackjack Game State Machine (Las Vegas Rules)
============================================================
Full implementation of the Blackjack game as a finite state machine
with the complete Las Vegas H17 ruleset:

  • Blackjack pays 3:2
  • Dealer hits Soft 17 (H17)
  • Dealer peeks for Blackjack on Ace or 10-value upcard
  • Double Down on any first two cards, including after split (DAS)
  • Re-splitting up to 4 total hands
  • Split Aces: one card each, no re-split, 10+A = 21 (not BJ)
  • Late Surrender (after dealer peek)
  • Insurance on Dealer Ace (pays 2:1, costs ½ bet)
"""

from enum import Enum, auto
from typing import List, Optional, Tuple, Dict
from deck import Shoe, RANK_NAMES, RANK_VALUES

# ── Constants ────────────────────────────────────────────────────────────────
MAX_SPLIT_HANDS = 4
BLACKJACK_PAYOUT = 1.5  # 3:2


class Action(Enum):
    """All possible player actions."""
    HIT = auto()
    STAND = auto()
    DOUBLE = auto()
    SPLIT = auto()
    SURRENDER = auto()
    INSURANCE = auto()


class GamePhase(Enum):
    """Game state machine phases."""
    BETTING = auto()         # Waiting for bet
    DEALING = auto()         # Cards being dealt
    INSURANCE = auto()       # Insurance offered (dealer shows Ace)
    PLAYER_TURN = auto()     # Player making decisions
    DEALER_TURN = auto()     # Dealer drawing cards
    PAYOUT = auto()          # Resolving bets
    ROUND_OVER = auto()      # Round complete


class Hand:
    """
    Represents a single Blackjack hand with cards and metadata.

    Calculates hard/soft totals, detects blackjack, bust, and pair status.
    Aces are counted as 11 (soft) when possible, downgraded to 1 (hard)
    when the total would exceed 21.
    """

    def __init__(self, bet: float = 0.0, is_split_hand: bool = False,
                 is_split_aces: bool = False):
        self.cards: List[int] = []        # List of rank indices
        self.suits: List[str] = []        # List of suits
        self.bet: float = bet
        self.is_split_hand: bool = is_split_hand
        self.is_split_aces: bool = is_split_aces
        self.is_stood: bool = False       # Player has stood
        self.is_doubled: bool = False     # Player has doubled
        self.is_surrendered: bool = False  # Player has surrendered
        self.is_complete: bool = False    # Hand is finished (stood/bust/21/etc.)

    def add_card(self, rank: int, suit: str = None) -> None:
        """Add a card (rank index 0-9) to the hand."""
        import random
        self.cards.append(rank)
        if suit is None:
            suit = random.choice(["♠", "♥", "♦", "♣"])
        self.suits.append(suit)
        # Auto-complete conditions
        if self.is_busted or self.total == 21:
            self.is_complete = True
        # Split aces get exactly one card, then auto-complete
        if self.is_split_aces and len(self.cards) == 2:
            self.is_complete = True

    @property
    def total(self) -> int:
        """
        Best non-busted total, or the busted total if unavoidable.

        Aces (rank 0) are initially counted as 11. Each ace is downgraded
        to 1 if the total exceeds 21, until no more aces can be downgraded.

        Formula:
            raw_total = sum(RANK_VALUES[r] for r in cards)
            while raw_total > 21 and aces_as_11 > 0:
                raw_total -= 10  # downgrade one ace from 11 to 1
        """
        raw = sum(RANK_VALUES[r] for r in self.cards)
        aces = sum(1 for r in self.cards if r == 0)
        while raw > 21 and aces > 0:
            raw -= 10
            aces -= 1
        return raw

    @property
    def is_soft(self) -> bool:
        """
        True if the hand contains an Ace counted as 11.

        A hand is soft when:
          - It contains at least one Ace, AND
          - Using at least one Ace as 11 doesn't bust
        """
        raw = sum(RANK_VALUES[r] for r in self.cards)
        aces = sum(1 for r in self.cards if r == 0)
        while raw > 21 and aces > 0:
            raw -= 10
            aces -= 1
        return aces > 0

    @property
    def is_blackjack(self) -> bool:
        """
        Natural Blackjack: exactly 2 cards totaling 21, NOT from a split.

        Split hands that reach 21 with two cards are counted as 21, not BJ.
        """
        return (
            len(self.cards) == 2
            and self.total == 21
            and not self.is_split_hand
        )

    @property
    def is_busted(self) -> bool:
        """True if total exceeds 21."""
        return self.total > 21

    @property
    def is_pair(self) -> bool:
        """True if hand has exactly 2 cards of the same rank."""
        return len(self.cards) == 2 and self.cards[0] == self.cards[1]

    @property
    def num_cards(self) -> int:
        return len(self.cards)

    def display(self, hide_first: bool = False) -> str:
        """
        String representation of the hand.

        Args:
            hide_first: If True, show first card as '??' (for dealer hidden card).
        """
        if not self.cards:
            return "[ ]"
        card_strs = []
        for i, rank in enumerate(self.cards):
            if i == 0 and hide_first:
                card_strs.append("🂠")
            else:
                card_strs.append(RANK_NAMES[rank])
        return " ".join(card_strs)

    def display_total(self, hide_first: bool = False) -> str:
        """Display total, considering hidden cards."""
        if hide_first and len(self.cards) >= 2:
            # Show only the visible card's value
            visible_rank = self.cards[1]  # second card is the upcard
            return f"({RANK_VALUES[visible_rank]})"
        soft_str = " (soft)" if self.is_soft else ""
        return f"({self.total}{soft_str})"

    def copy(self) -> "Hand":
        """Deep copy of the hand."""
        new = Hand(
            bet=self.bet,
            is_split_hand=self.is_split_hand,
            is_split_aces=self.is_split_aces,
        )
        new.cards = self.cards.copy()
        new.suits = self.suits.copy()
        new.is_stood = self.is_stood
        new.is_doubled = self.is_doubled
        new.is_surrendered = self.is_surrendered
        new.is_complete = self.is_complete
        return new

    def __repr__(self) -> str:
        return f"Hand({self.display()} = {self.total}, bet={self.bet})"


class BlackjackGame:
    """
    Full Blackjack game state machine implementing Las Vegas H17 rules.

    Manages the complete lifecycle of a round: betting, dealing, player
    decisions (including splits), dealer play, and payout resolution.

    Attributes:
        shoe (Shoe): The current shoe being dealt from.
        phase (GamePhase): Current state of the game.
        dealer_hand (Hand): The dealer's hand.
        player_hands (List[Hand]): Player's hand(s) (multiple if split).
        active_hand_idx (int): Index of the hand currently being played.
        bankroll (float): Player's current bankroll.
        insurance_bet (float): Insurance side-bet amount (0 if not taken).
    """

    def __init__(self, shoe: Shoe, bankroll: float = 10000.0):
        self.shoe = shoe
        self.bankroll = bankroll
        self.min_bet = 10.0
        self.max_bet = 500.0

        # Round state
        self.phase = GamePhase.BETTING
        self.dealer_hand = Hand()
        self.player_hands: List[Hand] = []
        self.active_hand_idx: int = 0
        self.insurance_bet: float = 0.0
        self.round_results: List[Dict] = []
        self.insurance_offered: bool = False
        self.insurance_resolved: bool = False

    @property
    def active_hand(self) -> Optional[Hand]:
        """The hand currently being played, or None if no active hands."""
        if 0 <= self.active_hand_idx < len(self.player_hands):
            return self.player_hands[self.active_hand_idx]
        return None

    @property
    def dealer_upcard(self) -> Optional[int]:
        """Dealer's visible card (second card dealt to dealer)."""
        if len(self.dealer_hand.cards) >= 2:
            return self.dealer_hand.cards[1]
        return None

    @property
    def dealer_hole_card(self) -> Optional[int]:
        """Dealer's hidden card (first card dealt)."""
        if len(self.dealer_hand.cards) >= 1:
            return self.dealer_hand.cards[0]
        return None

    def place_bet(self, amount: float) -> Tuple[bool, str]:
        """
        Place a bet to start a new round.

        Args:
            amount: Bet amount (clamped to [min_bet, max_bet]).

        Returns:
            (success, message) tuple.
        """
        if self.phase != GamePhase.BETTING:
            return False, "Cannot bet right now."

        amount = max(self.min_bet, min(amount, self.max_bet, self.bankroll))
        if amount > self.bankroll:
            return False, f"Insufficient bankroll (${self.bankroll:.2f})."

        self.bankroll -= amount
        self.player_hands = [Hand(bet=amount)]
        self.active_hand_idx = 0
        self.insurance_bet = 0.0
        self.insurance_offered = False
        self.insurance_resolved = False
        self.round_results = []
        self.phase = GamePhase.DEALING
        return True, f"Bet ${amount:.2f} placed."

    def deal(self) -> Tuple[str, bool]:
        """
        Deal initial cards: Player, Dealer, Player, Dealer.
        Check for dealer peek and natural blackjack.

        Returns:
            (message, round_over): Description of deal result and whether
            the round ended immediately (dealer or player BJ).
        """
        if self.phase != GamePhase.DEALING:
            return "Cannot deal right now.", False

        player = self.player_hands[0]

        # Deal: Player, Dealer, Player, Dealer
        player.add_card(self.shoe.deal_card())
        self.dealer_hand = Hand()
        self.dealer_hand.add_card(self.shoe.deal_card())  # hole card
        player.add_card(self.shoe.deal_card())
        self.dealer_hand.add_card(self.shoe.deal_card())  # upcard

        upcard = self.dealer_upcard

        # ── Dealer Peek Logic ────────────────────────────────────────────
        # Dealer peeks for BJ when showing Ace or 10-value
        dealer_has_bj = self.dealer_hand.is_blackjack

        # If dealer shows Ace, offer insurance first
        if upcard == 0:  # Ace
            self.insurance_offered = True
            if dealer_has_bj:
                # Will resolve after insurance decision
                self.phase = GamePhase.INSURANCE
                return "Dealer shows Ace. Insurance?", False

            self.phase = GamePhase.INSURANCE
            return "Dealer shows Ace. Insurance?", False

        # Dealer peeks on 10-value upcard
        if upcard == 9:  # 10-value (index 9)
            if dealer_has_bj:
                # Dealer has blackjack — resolve immediately
                self.phase = GamePhase.PAYOUT
                return self._resolve_dealer_blackjack(), True

        # ── Check Player Blackjack ───────────────────────────────────────
        if player.is_blackjack:
            # Player has natural BJ, dealer doesn't — immediate payout
            self.phase = GamePhase.PAYOUT
            winnings = player.bet * BLACKJACK_PAYOUT
            self.bankroll += player.bet + winnings
            self.round_results = [{
                "hand_idx": 0,
                "result": "BLACKJACK",
                "payout": winnings,
            }]
            self.phase = GamePhase.ROUND_OVER
            return f"Blackjack! You win ${winnings:.2f}!", True

        # Normal play
        self.phase = GamePhase.PLAYER_TURN
        return "Cards dealt. Your turn.", False

    def take_insurance(self, accept: bool) -> str:
        """
        Resolve insurance offer.

        Insurance costs ½ the original bet. Pays 2:1 if dealer has BJ.
        """
        if self.phase != GamePhase.INSURANCE:
            return "Insurance not available."

        player = self.player_hands[0]

        if accept:
            ins_cost = player.bet / 2.0
            if ins_cost > self.bankroll:
                ins_cost = self.bankroll
            self.bankroll -= ins_cost
            self.insurance_bet = ins_cost

        self.insurance_offered = False

        # Now check dealer blackjack
        if self.dealer_hand.is_blackjack:
            msg = self._resolve_dealer_blackjack()
            return msg

        # No dealer BJ — insurance bet lost if taken
        if self.insurance_bet > 0:
            self.round_results.append({
                "result": "INSURANCE_LOST",
                "payout": -self.insurance_bet,
            })

        # Check player blackjack
        if player.is_blackjack:
            winnings = player.bet * BLACKJACK_PAYOUT
            self.bankroll += player.bet + winnings
            self.round_results.append({
                "hand_idx": 0,
                "result": "BLACKJACK",
                "payout": winnings,
            })
            self.phase = GamePhase.ROUND_OVER
            return f"No dealer BJ. {'Insurance lost. ' if self.insurance_bet > 0 else ''}Blackjack! +${winnings:.2f}"

        self.phase = GamePhase.PLAYER_TURN
        ins_msg = "Insurance lost. " if self.insurance_bet > 0 else ""
        return f"No dealer Blackjack. {ins_msg}Your turn."

    def _resolve_dealer_blackjack(self) -> str:
        """Handle the case where dealer has a natural blackjack."""
        player = self.player_hands[0]
        msg_parts = ["Dealer has Blackjack!"]

        # Insurance payout (2:1)
        if self.insurance_bet > 0:
            ins_win = self.insurance_bet * 2
            self.bankroll += self.insurance_bet + ins_win
            msg_parts.append(f"Insurance pays ${ins_win:.2f}.")
            self.round_results.append({
                "result": "INSURANCE_WON",
                "payout": ins_win,
            })

        # Player also has BJ → push
        if player.is_blackjack:
            self.bankroll += player.bet  # return original bet
            msg_parts.append("Push on Blackjack.")
            self.round_results.append({
                "hand_idx": 0,
                "result": "PUSH",
                "payout": 0.0,
            })
        else:
            # Player loses
            msg_parts.append(f"You lose ${player.bet:.2f}.")
            self.round_results.append({
                "hand_idx": 0,
                "result": "LOSE",
                "payout": -player.bet,
            })

        self.phase = GamePhase.ROUND_OVER
        return " ".join(msg_parts)

    def available_actions(self) -> List[Action]:
        """
        Determine all legal actions for the current hand state.

        Rules applied:
          - Hit: always (unless hand is complete)
          - Stand: always
          - Double: first two cards only (including after split = DAS)
          - Split: pair with exactly 2 cards, total hands < MAX_SPLIT_HANDS,
                   not split aces (split aces cannot be re-split)
          - Surrender: first two cards, original hand only (not split),
                       after dealer peek (Late Surrender)
          - Insurance: handled separately in INSURANCE phase
        """
        if self.phase == GamePhase.INSURANCE:
            return [Action.INSURANCE]

        if self.phase != GamePhase.PLAYER_TURN:
            return []

        hand = self.active_hand
        if hand is None or hand.is_complete:
            return []

        actions = [Action.HIT, Action.STAND]

        # Double Down: any first two cards (including after split = DAS)
        if hand.num_cards == 2 and not hand.is_split_aces:
            if hand.bet <= self.bankroll:  # Need funds to double
                actions.append(Action.DOUBLE)

        # Split: pair, not split aces, under max hands
        if (hand.is_pair
                and len(self.player_hands) < MAX_SPLIT_HANDS
                and not hand.is_split_aces
                and hand.bet <= self.bankroll):  # Need funds for split
            actions.append(Action.SPLIT)

        # Late Surrender: first two cards of original hand only (not split)
        if (hand.num_cards == 2
                and not hand.is_split_hand
                and len(self.player_hands) == 1):
            actions.append(Action.SURRENDER)

        return actions

    def player_action(self, action: Action) -> Tuple[str, bool]:
        """
        Execute a player action on the active hand.

        Args:
            action: The action to take.

        Returns:
            (message, round_over): Result description and whether round ended.
        """
        hand = self.active_hand
        if hand is None:
            return "No active hand.", False

        if action == Action.HIT:
            return self._do_hit(hand)
        elif action == Action.STAND:
            return self._do_stand(hand)
        elif action == Action.DOUBLE:
            return self._do_double(hand)
        elif action == Action.SPLIT:
            return self._do_split(hand)
        elif action == Action.SURRENDER:
            return self._do_surrender(hand)
        else:
            return f"Invalid action: {action}", False

    def _do_hit(self, hand: Hand) -> Tuple[str, bool]:
        """Draw one card. If bust or 21, auto-complete."""
        card = self.shoe.deal_card()
        hand.add_card(card)

        if hand.is_busted:
            return self._advance_hand(f"Drew {RANK_NAMES[card]}. Bust! ({hand.total})")
        elif hand.total == 21:
            return self._advance_hand(f"Drew {RANK_NAMES[card]}. 21!")
        else:
            return f"Drew {RANK_NAMES[card]}. Total: {hand.total}{'(soft)' if hand.is_soft else ''}.", False

    def _do_stand(self, hand: Hand) -> Tuple[str, bool]:
        """Stand on current hand, move to next hand or dealer."""
        hand.is_stood = True
        hand.is_complete = True
        return self._advance_hand(f"Stand on {hand.total}.")

    def _do_double(self, hand: Hand) -> Tuple[str, bool]:
        """
        Double Down: double the bet, take exactly one card, then stand.
        """
        if hand.num_cards != 2:
            return "Can only double on first two cards.", False

        # Deduct additional bet
        self.bankroll -= hand.bet
        hand.bet *= 2
        hand.is_doubled = True

        # One card only
        card = self.shoe.deal_card()
        hand.add_card(card)
        hand.is_stood = True
        hand.is_complete = True

        if hand.is_busted:
            msg = f"Doubled. Drew {RANK_NAMES[card]}. Bust! ({hand.total})"
        else:
            msg = f"Doubled. Drew {RANK_NAMES[card]}. Total: {hand.total}."
        return self._advance_hand(msg)

    def _do_split(self, hand: Hand) -> Tuple[str, bool]:
        """
        Split a pair into two separate hands.

        Rules:
          - Each new hand gets one of the pair cards + one new card
          - Split Aces: one card each, auto-complete, cannot re-split
          - Other pairs: full play allowed, re-split up to 4 hands total
          - DAS allowed: can double after split
        """
        if not hand.is_pair:
            return "Cannot split — not a pair.", False
        if len(self.player_hands) >= MAX_SPLIT_HANDS:
            return "Maximum splits reached.", False

        split_rank = hand.cards[0]
        split_suit1 = hand.suits[0] if len(hand.suits) > 0 else "♠"
        split_suit2 = hand.suits[1] if len(hand.suits) > 1 else "♥"
        is_aces = (split_rank == 0)

        # Deduct bet for new hand
        self.bankroll -= hand.bet / 2  # Original hand already paid; new hand needs bet
        # Actually: each split hand has the original bet amount
        new_bet = hand.bet / 2  # Wait, let me reconsider...

        # The original bet stays. We need a new bet equal to original for the new hand.
        # hand.bet is the original bet. We need to put up another hand.bet for the 2nd hand.
        original_bet_per_hand = hand.bet  # This is what was bet on the original hand
        self.bankroll -= original_bet_per_hand  # Additional bet for second hand

        # Remove second card from original hand
        hand.cards = [split_rank]
        hand.suits = [split_suit1]
        hand.is_split_hand = True
        hand.is_split_aces = is_aces
        hand.is_complete = False
        hand.is_stood = False

        # Create new hand with the other card
        new_hand = Hand(
            bet=original_bet_per_hand,
            is_split_hand=True,
            is_split_aces=is_aces,
        )
        new_hand.cards = [split_rank]
        new_hand.suits = [split_suit2]

        # Insert new hand right after the current one
        self.player_hands.insert(self.active_hand_idx + 1, new_hand)

        # Deal one card to first split hand
        card1 = self.shoe.deal_card()
        hand.add_card(card1)

        # Deal one card to second split hand
        card2 = self.shoe.deal_card()
        new_hand.add_card(card2)

        msg = f"Split {RANK_NAMES[split_rank]}s. "

        # If split aces, both hands auto-complete (one card only)
        if is_aces:
            msg += f"Hand 1: {hand.display()} ({hand.total}). Hand 2: {new_hand.display()} ({new_hand.total}). "
            # Check if first hand needs advancing
            if hand.is_complete and new_hand.is_complete:
                # Both done, advance past both
                return self._advance_to_dealer(msg + "Split aces complete.")
            elif hand.is_complete:
                self.active_hand_idx += 1
                return msg + "Playing hand 2.", False

        # For non-ace splits, continue playing first hand
        if hand.is_complete:
            return self._advance_hand(msg + f"Hand 1 complete ({hand.total}).")

        return msg + f"Playing hand 1: {hand.display()} ({hand.total}).", False

    def _do_surrender(self, hand: Hand) -> Tuple[str, bool]:
        """
        Late Surrender: forfeit half the bet, hand is over.
        Player gets back half their original wager.
        """
        hand.is_surrendered = True
        hand.is_complete = True
        refund = hand.bet / 2.0
        self.bankroll += refund
        self.round_results.append({
            "hand_idx": self.active_hand_idx,
            "result": "SURRENDER",
            "payout": -refund,  # Net loss is half the bet
        })
        return self._advance_hand(f"Surrendered. Returned ${refund:.2f}.")

    def _advance_hand(self, msg: str) -> Tuple[str, bool]:
        """
        Move to the next incomplete hand, or to dealer turn if all done.
        """
        # Find next incomplete hand
        while self.active_hand_idx < len(self.player_hands):
            if not self.player_hands[self.active_hand_idx].is_complete:
                return msg, False
            self.active_hand_idx += 1

        # All player hands complete — move to dealer
        return self._advance_to_dealer(msg)

    def _advance_to_dealer(self, msg: str) -> Tuple[str, bool]:
        """Move game to dealer turn phase."""
        # Check if all hands busted or surrendered
        all_done = all(
            h.is_busted or h.is_surrendered for h in self.player_hands
        )
        if all_done:
            self.phase = GamePhase.PAYOUT
            return self.resolve(msg)

        self.phase = GamePhase.DEALER_TURN
        dealer_msg = self.play_dealer()
        self.phase = GamePhase.PAYOUT
        return self.resolve(msg + " " + dealer_msg)

    def play_dealer(self) -> str:
        """
        Dealer draws cards according to H17 rule.

        Rule: Dealer hits on soft 17 (H17), stands on hard 17+.

        The dealer must hit when:
          - Total < 17, OR
          - Total == 17 AND hand is soft (Soft 17 = Ace + 6)

        The dealer stands when:
          - Total >= 18, OR
          - Total == 17 AND hand is hard
        """
        draws = []
        while True:
            total = self.dealer_hand.total
            soft = self.dealer_hand.is_soft

            # H17: hit on soft 17, stand on hard 17+
            if total > 17:
                break
            if total == 17 and not soft:
                break
            # Hit
            card = self.shoe.deal_card()
            self.dealer_hand.add_card(card)
            draws.append(RANK_NAMES[card])

        if draws:
            draw_str = ", ".join(draws)
            return f"Dealer draws: {draw_str}. Dealer total: {self.dealer_hand.total}."
        return f"Dealer stands on {self.dealer_hand.total}."

    def resolve(self, prefix_msg: str = "") -> Tuple[str, bool]:
        """
        Resolve all bets and calculate payouts.

        Payout rules:
          - Blackjack: 3:2 (already handled in deal)
          - Win: 1:1
          - Push: bet returned
          - Lose: bet lost
          - Surrender: half bet returned (already handled)
          - Bust: bet lost
        """
        dealer_total = self.dealer_hand.total
        dealer_busted = self.dealer_hand.is_busted
        messages = [prefix_msg] if prefix_msg else []

        for i, hand in enumerate(self.player_hands):
            # Skip already-resolved hands (surrender)
            if hand.is_surrendered:
                continue

            if hand.is_busted:
                self.round_results.append({
                    "hand_idx": i,
                    "result": "BUST",
                    "payout": -hand.bet,
                })
                messages.append(f"Hand {i+1}: Bust. -${hand.bet:.2f}")
                continue

            player_total = hand.total

            if dealer_busted or player_total > dealer_total:
                # Player wins
                winnings = hand.bet
                self.bankroll += hand.bet + winnings
                self.round_results.append({
                    "hand_idx": i,
                    "result": "WIN",
                    "payout": winnings,
                })
                messages.append(f"Hand {i+1}: Win! +${winnings:.2f}")

            elif player_total == dealer_total:
                # Push
                self.bankroll += hand.bet
                self.round_results.append({
                    "hand_idx": i,
                    "result": "PUSH",
                    "payout": 0.0,
                })
                messages.append(f"Hand {i+1}: Push.")

            else:
                # Dealer wins
                self.round_results.append({
                    "hand_idx": i,
                    "result": "LOSE",
                    "payout": -hand.bet,
                })
                messages.append(f"Hand {i+1}: Lose. -${hand.bet:.2f}")

        self.phase = GamePhase.ROUND_OVER
        return " ".join(messages), True

    def new_round(self) -> str:
        """Reset for a new round. Check if shoe needs reshuffling."""
        reshuffle_msg = ""
        if self.shoe.needs_reshuffle:
            self.shoe.reset()
            reshuffle_msg = " Shoe reshuffled!"

        self.phase = GamePhase.BETTING
        self.dealer_hand = Hand()
        self.player_hands = []
        self.active_hand_idx = 0
        self.insurance_bet = 0.0
        self.insurance_offered = False
        self.insurance_resolved = False
        self.round_results = []

        return f"New round.{reshuffle_msg} Bankroll: ${self.bankroll:.2f}. Place your bet."

    def get_state_summary(self) -> Dict:
        """Return a summary of the current game state for the UI."""
        return {
            "phase": self.phase.name,
            "bankroll": self.bankroll,
            "dealer": {
                "cards": self.dealer_hand.display(
                    hide_first=(self.phase in (GamePhase.PLAYER_TURN, GamePhase.INSURANCE))
                ),
                "total": self.dealer_hand.display_total(
                    hide_first=(self.phase in (GamePhase.PLAYER_TURN, GamePhase.INSURANCE))
                ),
                "upcard": RANK_NAMES[self.dealer_upcard] if self.dealer_upcard is not None else "?",
            },
            "player_hands": [
                {
                    "cards": h.display(),
                    "total": h.total,
                    "is_soft": h.is_soft,
                    "bet": h.bet,
                    "status": (
                        "SURRENDERED" if h.is_surrendered
                        else "BUST" if h.is_busted
                        else "BLACKJACK" if h.is_blackjack
                        else "STOOD" if h.is_stood
                        else "ACTIVE" if i == self.active_hand_idx
                        else "WAITING"
                    ),
                }
                for i, h in enumerate(self.player_hands)
            ],
            "active_hand": self.active_hand_idx,
            "actions": [a.name for a in self.available_actions()],
            "insurance_offered": self.phase == GamePhase.INSURANCE,
            "results": self.round_results,
        }
