"""
app.py — Gradio Interactive Blackjack Frontend
=================================================
Two-panel layout:
  Left:  Game Table — cards, actions, bet controls
  Right: AI HUD — counting, probabilities, EV analysis

State is persisted via gr.State() across UI interactions.
All game parameters (bankroll, min/max bet, decks) are user-configurable.
"""

import gradio as gr
import numpy as np
import sys
import os
import random
from typing import Dict, Any, Optional, Tuple

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from deck import Shoe, RANK_NAMES
from engine import BlackjackGame, Action, GamePhase, Hand
from math_engine import HiLoCounter, ProbabilityEngine
from ai_advisor import EVCalculator, KellyCriterion
from mass_sim import run_simulation

# ── Card Display (Unicode/Emoji) ─────────────────────────────────────────────
CARD_DISPLAY = {
    0: "🂡",  # Ace
    1: "🂢",  # 2
    2: "🂣",  # 3
    3: "🂤",  # 4
    4: "🂥",  # 5
    5: "🂦",  # 6
    6: "🂧",  # 7
    7: "🂨",  # 8
    8: "🂩",  # 9
    9: "🂪",  # 10
}

SUIT_EMOJI = ["♠", "♥", "♦", "♣"]


def _card_str(rank: int) -> str:
    """Render a card rank as a nice string."""
    return RANK_NAMES[rank]


def _cards_display(hand, hide_first: bool = False) -> str:
    """Render a hand of cards as HTML."""
    if not hand or not hasattr(hand, 'cards') or not hand.cards:
        return ""
    html = []
    for i, rank in enumerate(hand.cards):
        if i == 0 and hide_first:
            html.append('<div class="card card-hidden"></div>')
        else:
            rank_name = RANK_NAMES[rank]
            suit = hand.suits[i] if hasattr(hand, 'suits') and i < len(hand.suits) else random.choice(SUIT_EMOJI)
            color_class = "card-red" if suit in ["♥", "♦"] else "card-black"
            html.append(f'''
            <div class="card {color_class}">
                <div class="card-topleft">{rank_name}<br>{suit}</div>
                <div class="card-center">{suit}</div>
                <div class="card-bottomright">{rank_name}<br>{suit}</div>
            </div>
            ''')
    return "".join(html)


def _format_ev_table(evs: Dict[str, float], optimal: Optional[str] = None) -> str:
    """Format EV results as an aligned text table with the optimal highlighted."""
    if not evs:
        return "No actions available."

    lines = []
    sorted_evs = sorted(evs.items(), key=lambda x: x[1], reverse=True)

    for action, ev in sorted_evs:
        marker = "  ★ OPTIMAL" if action == optimal else ""
        bar_width = int(max(0, min(20, (ev + 1) * 10)))  # Scale -1..+1 to 0..20
        bar = "█" * bar_width + "░" * (20 - bar_width)
        sign = "+" if ev >= 0 else ""
        lines.append(f"  {action:<12} {sign}{ev:>8.4f}  {bar}{marker}")

    return "\n".join(lines)


def _format_dealer_probs(probs: Dict[str, float]) -> str:
    """Format dealer outcome probabilities as a table."""
    if not probs:
        return "Calculating..."

    lines = []
    for outcome in ["17", "18", "19", "20", "21", "bust"]:
        p = probs.get(outcome, 0.0)
        bar = "█" * int(p * 40) + "░" * (40 - int(p * 40))
        label = "BUST" if outcome == "bust" else outcome
        lines.append(f"  {label:<6} {p*100:>6.2f}%  {bar}")

    return "\n".join(lines)


# ── Initialize Default State ─────────────────────────────────────────────────

def create_initial_state(
    bankroll: float = 10000.0,
    min_bet: float = 10.0,
    max_bet: float = 500.0,
    num_decks: int = 6,
) -> Dict[str, Any]:
    """Create a fresh game state dictionary."""
    shoe = Shoe(num_decks=num_decks)
    game = BlackjackGame(shoe, bankroll=bankroll)
    game.min_bet = min_bet
    game.max_bet = max_bet
    counter = HiLoCounter()
    EVCalculator.clear_cache()

    return {
        "shoe": shoe,
        "game": game,
        "counter": counter,
        "bankroll": bankroll,
        "min_bet": min_bet,
        "max_bet": max_bet,
        "num_decks": num_decks,
        "bet_amount": min_bet,
        "message": f"Welcome! ${bankroll:,.0f} bankroll, {num_decks}-deck shoe. Place your bet.",
        "round_count": 0,
        "history": [],
    }


def _observe_dealt_cards(state: Dict, old_shoe_cards: np.ndarray) -> None:
    """Update the Hi-Lo counter for any cards dealt since last check."""
    diff = old_shoe_cards - state["shoe"].cards
    for rank in range(10):
        for _ in range(max(0, int(diff[rank]))):
            state["counter"].observe_card(rank)


# ── Core Game Actions ────────────────────────────────────────────────────────

def action_new_game(
    state: Dict,
    bankroll: float,
    min_bet: float,
    max_bet: float,
    num_decks: int,
) -> Tuple:
    """Start a completely new game with new settings."""
    state = create_initial_state(
        bankroll=bankroll,
        min_bet=min_bet,
        max_bet=max_bet,
        num_decks=int(num_decks),
    )
    return _build_outputs(state)


def action_deal(state: Dict, bet_amount: float) -> Tuple:
    """Place bet and deal cards."""
    game = state["game"]
    shoe = state["shoe"]

    if game.phase == GamePhase.ROUND_OVER:
        msg = game.new_round()
        state["message"] = msg
        # Check for reshuffle — counter resets
        if shoe.total_remaining == shoe.total_cards:
            state["counter"].reset()
            EVCalculator.clear_cache()

    if game.phase != GamePhase.BETTING:
        state["message"] = "Cannot deal right now. Finish the current hand."
        return _build_outputs(state)

    bet = max(state["min_bet"], min(bet_amount, state["max_bet"]))
    state["bet_amount"] = bet

    old_cards = shoe.cards.copy()
    success, msg = game.place_bet(bet)
    if not success:
        state["message"] = msg
        return _build_outputs(state)

    deal_msg, round_over = game.deal()
    _observe_dealt_cards(state, old_cards)

    state["message"] = deal_msg
    state["round_count"] += 1

    if round_over:
        state["bankroll"] = game.bankroll

    return _build_outputs(state)


def action_hit(state: Dict) -> Tuple:
    """Player hits."""
    game = state["game"]
    shoe = state["shoe"]
    old_cards = shoe.cards.copy()

    msg, round_over = game.player_action(Action.HIT)
    _observe_dealt_cards(state, old_cards)

    state["message"] = msg
    state["bankroll"] = game.bankroll
    return _build_outputs(state)


def action_stand(state: Dict) -> Tuple:
    """Player stands."""
    game = state["game"]
    shoe = state["shoe"]
    old_cards = shoe.cards.copy()

    msg, round_over = game.player_action(Action.STAND)
    _observe_dealt_cards(state, old_cards)

    state["message"] = msg
    state["bankroll"] = game.bankroll
    return _build_outputs(state)


def action_double(state: Dict) -> Tuple:
    """Player doubles down."""
    game = state["game"]
    shoe = state["shoe"]
    old_cards = shoe.cards.copy()

    msg, round_over = game.player_action(Action.DOUBLE)
    _observe_dealt_cards(state, old_cards)

    state["message"] = msg
    state["bankroll"] = game.bankroll
    return _build_outputs(state)


def action_split(state: Dict) -> Tuple:
    """Player splits."""
    game = state["game"]
    shoe = state["shoe"]
    old_cards = shoe.cards.copy()

    msg, round_over = game.player_action(Action.SPLIT)
    _observe_dealt_cards(state, old_cards)

    state["message"] = msg
    state["bankroll"] = game.bankroll
    return _build_outputs(state)


def action_surrender(state: Dict) -> Tuple:
    """Player surrenders."""
    game = state["game"]
    shoe = state["shoe"]
    old_cards = shoe.cards.copy()

    msg, round_over = game.player_action(Action.SURRENDER)
    _observe_dealt_cards(state, old_cards)

    state["message"] = msg
    state["bankroll"] = game.bankroll
    return _build_outputs(state)


def action_insurance_yes(state: Dict) -> Tuple:
    """Accept insurance."""
    game = state["game"]
    msg = game.take_insurance(True)
    state["message"] = msg
    state["bankroll"] = game.bankroll
    return _build_outputs(state)


def action_insurance_no(state: Dict) -> Tuple:
    """Decline insurance."""
    game = state["game"]
    msg = game.take_insurance(False)
    state["message"] = msg
    state["bankroll"] = game.bankroll
    return _build_outputs(state)


# ── Build UI Output ──────────────────────────────────────────────────────────

def _build_outputs(state: Dict) -> Tuple:
    """
    Build all UI component values from current state.

    Returns tuple of:
      (state, message, dealer_display, player_display,
       hud_counting, hud_dealer_probs, hud_ev_table, hud_kelly,
       btn_deal, btn_hit, btn_stand, btn_double, btn_split,
       btn_surrender, btn_ins_yes, btn_ins_no, bankroll_display)
    """
    game = state["game"]
    shoe = state["shoe"]
    counter = state["counter"]
    phase = game.phase

    # ── Message ──────────────────────────────────────────────────────
    message = state["message"]

    # ── Dealer Display ───────────────────────────────────────────────
    hide_dealer = phase in (GamePhase.PLAYER_TURN, GamePhase.INSURANCE)
    if game.dealer_hand.cards:
        dealer_cards = _cards_display(game.dealer_hand, hide_first=hide_dealer)
        if hide_dealer:
            dealer_total = f"Showing: {RANK_NAMES[game.dealer_upcard]}" if game.dealer_upcard is not None else "?"
        else:
            soft_str = " (soft)" if game.dealer_hand.is_soft else ""
            dealer_total = f"Total: {game.dealer_hand.total}{soft_str}"
        dealer_display = f'<div class="game-area-html"><h3 style="margin:0 0 10px 0;">🃏 DEALER</h3><div>{dealer_cards}</div><div style="margin-top:10px; font-size:16px;">{dealer_total}</div></div>'
    else:
        dealer_display = '<div class="game-area-html"><h3>🃏 DEALER</h3><p>Waiting for deal...</p></div>'

    # ── Player Display ───────────────────────────────────────────────
    if game.player_hands:
        player_parts = []
        for i, hand in enumerate(game.player_hands):
            is_active = (i == game.active_hand_idx and phase == GamePhase.PLAYER_TURN)
            active_class = "hand-active" if is_active else ""
            soft_str = " (soft)" if hand.is_soft else ""
            status = ""
            if hand.is_surrendered:
                status = " <span style='color:#f85149;'>[SURRENDERED]</span>"
            elif hand.is_busted:
                status = " <span style='color:#f85149;'>[BUST]</span>"
            elif hand.is_blackjack:
                status = " <span style='color:#c9a032;'>[BLACKJACK!]</span>"
            elif hand.is_stood:
                status = " <span style='color:#8b949e;'>[STOOD]</span>"

            cards_html = _cards_display(hand)
            player_parts.append(f'''
            <div class="hand-container {active_class}">
                <div style="margin-bottom:8px;"><strong>Hand {i+1} (${hand.bet:.0f})</strong>: {hand.total}{soft_str}{status}</div>
                <div>{cards_html}</div>
            </div>
            ''')
        player_display = f'<div class="game-area-html"><h3 style="margin:0 0 10px 0;">🂠 PLAYER</h3>{"".join(player_parts)}</div>'
    else:
        player_display = '<div class="game-area-html"><h3>🂠 PLAYER</h3><p>Place a bet to begin.</p></div>'

    # ── Bankroll Display ─────────────────────────────────────────────
    bankroll_display = f"${game.bankroll:,.2f}"

    # ── HUD: Card Counting ───────────────────────────────────────────
    tc = counter.true_count(shoe)
    rc = counter.running_count
    decks_rem = shoe.decks_remaining
    pen = shoe.penetration

    hud_counting = (
        f"📊  CARD COUNTING HUD\n"
        f"{'─' * 36}\n"
        f"  Running Count:   {rc:>+6d}\n"
        f"  True Count:      {tc:>+6.1f}\n"
        f"  Decks Remaining: {decks_rem:>6.1f}\n"
        f"  Shoe Penetration:{pen:>6.1%}\n"
        f"  Cards Dealt:     {shoe.cards_dealt:>6d} / {shoe.total_cards}"
    )

    # ── HUD: Dealer Bust Probability ─────────────────────────────────
    hud_dealer_probs = "📉  DEALER OUTCOME PROBABILITIES\n" + "─" * 36 + "\n"
    if game.dealer_upcard is not None and phase in (GamePhase.PLAYER_TURN, GamePhase.INSURANCE):
        try:
            probs = ProbabilityEngine.dealer_outcome_probabilities(
                game.dealer_upcard, shoe
            )
            hud_dealer_probs += _format_dealer_probs(probs)
            bust_pct = probs.get("bust", 0.0) * 100
            hud_dealer_probs += f"\n\n  ⚠  Dealer Bust: {bust_pct:.1f}%"
        except Exception as e:
            hud_dealer_probs += f"  (Error: {e})"
    elif phase == GamePhase.ROUND_OVER and game.dealer_hand.cards:
        final_total = game.dealer_hand.total
        if game.dealer_hand.is_busted:
            hud_dealer_probs += f"  Dealer BUSTED with {final_total}."
        else:
            hud_dealer_probs += f"  Dealer finished with {final_total}."
    else:
        hud_dealer_probs += "  Waiting for deal..."

    # ── HUD: EV Table ────────────────────────────────────────────────
    hud_ev_table = "🧠  EXPECTED VALUES (EV)\n" + "─" * 36 + "\n"
    if phase == GamePhase.PLAYER_TURN and game.active_hand is not None:
        try:
            evs = EVCalculator.all_evs(game, shoe)
            optimal = EVCalculator.optimal_action(evs)
            hud_ev_table += _format_ev_table(evs, optimal)
            if optimal:
                hud_ev_table += f"\n\n  🎯 Recommended: {optimal}"
        except Exception as e:
            hud_ev_table += f"  (Error computing EV: {e})"
    elif phase == GamePhase.INSURANCE:
        try:
            ins_ev = EVCalculator.ev_insurance(shoe)
            sign = "+" if ins_ev >= 0 else ""
            rec = "TAKE Insurance" if ins_ev > 0 else "DECLINE Insurance"
            hud_ev_table += f"  INSURANCE EV:  {sign}{ins_ev:.4f}\n\n  🎯 Recommended: {rec}"
        except Exception as e:
            hud_ev_table += f"  (Error: {e})"
    else:
        hud_ev_table += "  Waiting for player turn..."

    # ── HUD: Kelly Criterion ─────────────────────────────────────────
    bet_info = KellyCriterion.bet_info(tc, game.bankroll, state["min_bet"], state["max_bet"])
    edge_str = f"{bet_info['edge_pct']:+.2f}%"
    kelly_str = f"{bet_info['kelly_fraction']:.4f}"
    bet_rec = f"${bet_info['recommended_bet']:,.0f}"
    bet_mult = f"{bet_info['bet_multiple']:.1f}×"

    hud_kelly = (
        f"💰  KELLY CRITERION BET SIZING\n"
        f"{'─' * 36}\n"
        f"  Estimated Edge:  {edge_str:>10}\n"
        f"  Kelly Fraction:  {kelly_str:>10}\n"
        f"  Recommended Bet: {bet_rec:>10}\n"
        f"  Bet Multiple:    {bet_mult:>10} min"
    )

    # ── Button States ────────────────────────────────────────────────
    actions = game.available_actions()
    is_player_turn = (phase == GamePhase.PLAYER_TURN)
    is_betting = (phase in (GamePhase.BETTING, GamePhase.ROUND_OVER))
    is_insurance = (phase == GamePhase.INSURANCE)

    btn_deal = gr.Button(interactive=is_betting)
    btn_hit = gr.Button(interactive=is_player_turn and Action.HIT in actions)
    btn_stand = gr.Button(interactive=is_player_turn and Action.STAND in actions)
    btn_double = gr.Button(interactive=is_player_turn and Action.DOUBLE in actions)
    btn_split = gr.Button(interactive=is_player_turn and Action.SPLIT in actions)
    btn_surrender = gr.Button(interactive=is_player_turn and Action.SURRENDER in actions)
    btn_ins_yes = gr.Button(interactive=is_insurance, visible=is_insurance)
    btn_ins_no = gr.Button(interactive=is_insurance, visible=is_insurance)

    return (
        state,
        message,
        dealer_display,
        player_display,
        hud_counting,
        hud_dealer_probs,
        hud_ev_table,
        hud_kelly,
        btn_deal,
        btn_hit,
        btn_stand,
        btn_double,
        btn_split,
        btn_surrender,
        btn_ins_yes,
        btn_ins_no,
        bankroll_display,
    )


# ── Mass Simulation Handler ──────────────────────────────────────────────────

def run_mass_sim(
    num_hands: int,
    num_decks: int,
    bet_size: float,
    use_counting: bool,
    min_bet: float,
    max_bet: float,
) -> str:
    """Run mass simulation and return formatted results."""
    try:
        num_hands = int(num_hands)
        num_decks = int(num_decks)
        result = run_simulation(
            num_hands=num_hands,
            num_decks=num_decks,
            bet_size=bet_size,
            use_counting=use_counting,
            min_bet=min_bet,
            max_bet=max_bet,
        )
        return result.summary()
    except Exception as e:
        return f"Simulation Error: {e}"


# ── Build Gradio Interface ───────────────────────────────────────────────────

# ── CSS and Theme (module-level for Gradio 6.x launch()) ────────────────────

APP_THEME = gr.themes.Base(
    primary_hue="green",
    secondary_hue="blue",
    neutral_hue="gray",
)

def create_app():
    """Build and return the Gradio Blocks application."""

    # Custom CSS for premium dark theme
    custom_css = """
    /* ── Global Theme ──────────────────────────────── */
    .gradio-container {
        background: linear-gradient(135deg, #0a0e17 0%, #1a1f2e 50%, #0d1321 100%) !important;
        font-family: 'SF Mono', 'Fira Code', 'Cascadia Code', monospace !important;
        color: #e0e6f0 !important;
        min-height: 100vh;
    }

    /* ── Realistic Playing Cards ────────────────────── */
    .card {
        display: inline-block;
        width: 70px;
        height: 100px;
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #d1d5db;
        margin: 4px;
        position: relative;
        box-shadow: 2px 2px 6px rgba(0,0,0,0.4);
        color: #111827;
        font-family: 'Helvetica Neue', Arial, sans-serif;
        vertical-align: top;
        background-image: none;
        box-sizing: border-box;
    }
    .card-black, .card-black div { 
        color: #111827 !important; 
        -webkit-text-fill-color: #111827 !important;
    }
    .card-red, .card-red div { 
        color: #dc2626 !important; 
        -webkit-text-fill-color: #dc2626 !important;
    }
    .card-hidden {
        background-color: #1e3a8a;
        background-image: repeating-linear-gradient(45deg, transparent, transparent 5px, rgba(255,255,255,.1) 5px, rgba(255,255,255,.1) 10px);
        border: 2px solid #ffffff;
    }
    .card-topleft {
        position: absolute;
        top: 4px;
        left: 6px;
        font-size: 16px;
        font-weight: bold;
        line-height: 1.1;
        text-align: center;
    }
    .card-center {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 32px;
    }
    .card-bottomright {
        position: absolute;
        bottom: 4px;
        right: 6px;
        font-size: 16px;
        font-weight: bold;
        line-height: 1.1;
        text-align: center;
        transform: rotate(180deg);
    }
    .hand-container {
        margin-bottom: 12px;
        padding: 10px;
        border-radius: 8px;
        background: rgba(0,0,0,0.25);
    }
    .hand-active {
        box-shadow: 0 0 10px 2px #3b82f6;
        border: 1px solid #3b82f6;
        background: rgba(59,130,246,0.15);
    }
    .game-area-html {
        min-height: 180px;
        padding: 20px;
        background: linear-gradient(180deg, #0c2e1a 0%, #0a3d1f 50%, #0c2e1a 100%);
        border: 2px solid #1a5c35;
        border-radius: 12px;
        color: #e8f5e9;
        font-family: 'SF Mono', 'Fira Code', monospace;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.4), 0 0 20px rgba(26,92,53,0.15);
        margin-bottom: 16px;
    }
    .game-area-html h3 {
        color: #7ee787;
        margin-top: 0;
    }

    /* ── Card Table Felt Effect ────────────────────── */
    .hud-panel textarea {
        background: linear-gradient(180deg, #0d1117 0%, #161b22 50%, #0d1117 100%) !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        font-family: 'SF Mono', 'Fira Code', monospace !important;
        font-size: 15px !important;
        line-height: 1.6 !important;
        padding: 16px !important;
        box-shadow: inset 0 1px 4px rgba(0, 0, 0, 0.5),
                    0 0 15px rgba(56, 139, 253, 0.08) !important;
    }

    /* ── Action Buttons ────────────────────────────── */
    .action-btn button {
        background: linear-gradient(180deg, #238636 0%, #1a7f37 100%) !important;
        border: 1px solid #2ea043 !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        padding: 12px 20px !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 2px 8px rgba(35, 134, 54, 0.3) !important;
    }
    .action-btn button:hover:not(:disabled) {
        background: linear-gradient(180deg, #2ea043 0%, #238636 100%) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(46, 160, 67, 0.4) !important;
    }
    .action-btn button:disabled {
        background: #21262d !important;
        border-color: #30363d !important;
        color: #484f58 !important;
        box-shadow: none !important;
        cursor: not-allowed !important;
    }

    /* ── Deal Button (Gold) ────────────────────────── */
    .deal-btn button {
        background: linear-gradient(180deg, #c9a032 0%, #b8860b 100%) !important;
        border: 1px solid #d4a843 !important;
        box-shadow: 0 2px 12px rgba(201, 160, 50, 0.35) !important;
    }
    .deal-btn button:hover:not(:disabled) {
        background: linear-gradient(180deg, #d4a843 0%, #c9a032 100%) !important;
        box-shadow: 0 4px 20px rgba(212, 168, 67, 0.5) !important;
    }

    /* ── Surrender Button (Red) ────────────────────── */
    .surrender-btn button {
        background: linear-gradient(180deg, #da3633 0%, #b62324 100%) !important;
        border: 1px solid #f85149 !important;
        box-shadow: 0 2px 8px rgba(218, 54, 51, 0.3) !important;
    }
    .surrender-btn button:hover:not(:disabled) {
        box-shadow: 0 4px 16px rgba(248, 81, 73, 0.4) !important;
    }

    /* ── Insurance Buttons ─────────────────────────── */
    .ins-btn button {
        background: linear-gradient(180deg, #1f6feb 0%, #1a5cc7 100%) !important;
        border: 1px solid #388bfd !important;
        box-shadow: 0 2px 8px rgba(31, 111, 235, 0.3) !important;
    }

    /* ── Settings Panel ────────────────────────────── */
    .settings-panel {
        background: rgba(22, 27, 34, 0.8) !important;
        border: 1px solid #30363d !important;
        border-radius: 12px !important;
        padding: 12px !important;
    }

    /* ── Message Bar ───────────────────────────────── */
    .message-bar textarea {
        background: linear-gradient(90deg, #161b22 0%, #1c2333 100%) !important;
        border: 1px solid #388bfd44 !important;
        border-radius: 8px !important;
        color: #58a6ff !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        text-align: center !important;
        padding: 12px !important;
    }

    /* ── Bankroll Display ──────────────────────────── */
    .bankroll-display textarea {
        background: transparent !important;
        border: none !important;
        color: #7ee787 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
        text-align: center !important;
        text-shadow: 0 0 20px rgba(126, 231, 135, 0.3) !important;
    }

    /* ── Simulation Results ────────────────────────── */
    .sim-results textarea {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #7ee787 !important;
        font-family: 'SF Mono', 'Fira Code', monospace !important;
        font-size: 13px !important;
        border-radius: 8px !important;
    }

    /* ── Tab Styling ───────────────────────────────── */
    .tabs .tab-nav button {
        color: #8b949e !important;
        background: transparent !important;
        border-bottom: 2px solid transparent !important;
        font-weight: 600 !important;
    }
    .tabs .tab-nav button.selected {
        color: #58a6ff !important;
        border-bottom: 2px solid #58a6ff !important;
    }

    /* ── Number Inputs ─────────────────────────────── */
    input[type="number"] {
        background: #0d1117 !important;
        border: 1px solid #30363d !important;
        color: #c9d1d9 !important;
        border-radius: 6px !important;
    }

    /* ── Labels ────────────────────────────────────── */
    label {
        color: #8b949e !important;
        font-weight: 500 !important;
    }

    /* ── Scrollbar ─────────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0d1117; }
    ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }
    """

    with gr.Blocks(
        title="♠ Blackjack AI — Casino Simulator & EV Engine by Khoa Hoang",
    ) as app:

        # ── State ────────────────────────────────────────────────────
        state = gr.State(create_initial_state())

        # ── Header ───────────────────────────────────────────────────
        gr.Markdown(
            """
            # ♠ BLACKJACK AI
            ### Casino Simulator • Combinatoric Engine • EV Maximizer • by Khoa Hoang
            """,
        )

        with gr.Tabs() as tabs:
            # ══════════════════════════════════════════════════════════
            # TAB 1: GAME TABLE
            # ══════════════════════════════════════════════════════════
            with gr.TabItem("🃏 Game Table", id="game"):

                # ── Settings Row ─────────────────────────────────────
                with gr.Row(elem_classes=["settings-panel"]):
                    with gr.Column(scale=1):
                        inp_bankroll = gr.Number(
                            label="💰 Bankroll ($)",
                            value=10000,
                            minimum=100,
                            maximum=1000000,
                            step=100,
                        )
                    with gr.Column(scale=1):
                        inp_min_bet = gr.Number(
                            label="📉 Min Bet ($)",
                            value=10,
                            minimum=1,
                            maximum=10000,
                            step=1,
                        )
                    with gr.Column(scale=1):
                        inp_max_bet = gr.Number(
                            label="📈 Max Bet ($)",
                            value=500,
                            minimum=10,
                            maximum=100000,
                            step=10,
                        )
                    with gr.Column(scale=1):
                        inp_num_decks = gr.Number(
                            label="🂠 Number of Decks",
                            value=6,
                            minimum=1,
                            maximum=8,
                            step=1,
                        )
                    with gr.Column(scale=1):
                        btn_new_game = gr.Button(
                            "🔄  NEW GAME",
                            variant="secondary",
                            size="lg",
                        )

                msg_display = gr.Textbox(
                    show_label=False,
                    value="Welcome! Place your bet to begin.",
                    interactive=False,
                    lines=1,
                    elem_classes=["message-bar"],
                )

                with gr.Row():
                    # ── LEFT: Game Panel ─────────────────────────────
                    with gr.Column(scale=3):
                        # Bankroll
                        bankroll_display = gr.Textbox(
                            label="BANKROLL",
                            value="$10,000.00",
                            interactive=False,
                            lines=1,
                            elem_classes=["bankroll-display"],
                        )

                        # Dealer
                        dealer_display = gr.HTML(
                            value='<div class="game-area-html"><h3>🃏 DEALER</h3><p>Waiting for deal...</p></div>',
                            elem_classes=["game-panel-html"],
                        )

                        # Player
                        player_display = gr.HTML(
                            value='<div class="game-area-html"><h3>🂠 PLAYER</h3><p>Place a bet to begin.</p></div>',
                            elem_classes=["game-panel-html"],
                        )

                        # Bet + Deal
                        with gr.Row():
                            inp_bet = gr.Number(
                                label="Bet Amount ($)",
                                value=10,
                                minimum=1,
                                step=5,
                            )
                            btn_deal = gr.Button(
                                "🂡  DEAL",
                                variant="primary",
                                size="lg",
                                elem_classes=["deal-btn"],
                            )

                        # Action Buttons
                        with gr.Row():
                            btn_hit = gr.Button(
                                "👆  HIT",
                                interactive=False,
                                elem_classes=["action-btn"],
                            )
                            btn_stand = gr.Button(
                                "✋  STAND",
                                interactive=False,
                                elem_classes=["action-btn"],
                            )
                            btn_double = gr.Button(
                                "⏫  DOUBLE",
                                interactive=False,
                                elem_classes=["action-btn"],
                            )
                            btn_split = gr.Button(
                                "✂  SPLIT",
                                interactive=False,
                                elem_classes=["action-btn"],
                            )
                            btn_surrender = gr.Button(
                                "🏳  SURRENDER",
                                interactive=False,
                                elem_classes=["surrender-btn"],
                            )

                        # Insurance Buttons (hidden by default)
                        with gr.Row():
                            btn_ins_yes = gr.Button(
                                "🛡  TAKE INSURANCE",
                                visible=False,
                                elem_classes=["ins-btn"],
                            )
                            btn_ins_no = gr.Button(
                                "❌  NO INSURANCE",
                                visible=False,
                                elem_classes=["ins-btn"],
                            )

                    # ── RIGHT: AI HUD ────────────────────────────────
                    with gr.Column(scale=2):
                        hud_counting = gr.Textbox(
                            label="Card Counting HUD",
                            value="📊  CARD COUNTING HUD\n" + "─" * 36 + "\n  Waiting for game...",
                            interactive=False,
                            lines=7,
                            elem_classes=["hud-panel"],
                        )
                        hud_dealer_probs = gr.Textbox(
                            label="Dealer Probabilities",
                            value="📉  DEALER OUTCOME PROBABILITIES\n" + "─" * 36 + "\n  Waiting for deal...",
                            interactive=False,
                            lines=10,
                            elem_classes=["hud-panel"],
                        )
                        hud_ev_table = gr.Textbox(
                            label="Expected Values (EV)",
                            value="🧠  EXPECTED VALUES (EV)\n" + "─" * 36 + "\n  Waiting for player turn...",
                            interactive=False,
                            lines=10,
                            elem_classes=["hud-panel"],
                        )
                        hud_kelly = gr.Textbox(
                            label="Kelly Criterion",
                            value="💰  KELLY CRITERION BET SIZING\n" + "─" * 36 + "\n  Waiting...",
                            interactive=False,
                            lines=6,
                            elem_classes=["hud-panel"],
                        )

            # ══════════════════════════════════════════════════════════
            # TAB 2: MASS SIMULATION
            # ══════════════════════════════════════════════════════════
            with gr.TabItem("📊 Mass Simulation", id="sim"):
                gr.Markdown("### ⚡ Vectorized Mass Simulation Engine")
                gr.Markdown(
                    "Run 100K+ hands using NumPy vectorized operations. "
                    "All hands are dealt and resolved concurrently via matrix operations."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        sim_num_hands = gr.Number(
                            label="Number of Hands",
                            value=100000,
                            minimum=1000,
                            maximum=1000000,
                            step=10000,
                        )
                        sim_num_decks = gr.Number(
                            label="Number of Decks",
                            value=6,
                            minimum=1,
                            maximum=8,
                            step=1,
                        )
                        sim_bet_size = gr.Number(
                            label="Bet Size ($)",
                            value=10,
                            minimum=1,
                            step=5,
                        )
                        sim_counting = gr.Checkbox(
                            label="Use Hi-Lo Counting (Bet Variation)",
                            value=False,
                        )
                        sim_min_bet = gr.Number(
                            label="Min Bet ($, for counting)",
                            value=10,
                            minimum=1,
                        )
                        sim_max_bet = gr.Number(
                            label="Max Bet ($, for counting)",
                            value=500,
                            minimum=10,
                        )
                        btn_run_sim = gr.Button(
                            "🚀  RUN SIMULATION",
                            variant="primary",
                            size="lg",
                            elem_classes=["deal-btn"],
                        )

                    with gr.Column(scale=2):
                        sim_results = gr.Textbox(
                            label="Results",
                            value="Click 'RUN SIMULATION' to begin...",
                            interactive=False,
                            lines=25,
                            elem_classes=["sim-results"],
                        )

        # ── Output Components List ───────────────────────────────────
        outputs = [
            state,
            msg_display,
            dealer_display,
            player_display,
            hud_counting,
            hud_dealer_probs,
            hud_ev_table,
            hud_kelly,
            btn_deal,
            btn_hit,
            btn_stand,
            btn_double,
            btn_split,
            btn_surrender,
            btn_ins_yes,
            btn_ins_no,
            bankroll_display,
        ]

        # ── Wire Up Events ───────────────────────────────────────────
        btn_new_game.click(
            fn=action_new_game,
            inputs=[state, inp_bankroll, inp_min_bet, inp_max_bet, inp_num_decks],
            outputs=outputs,
        )

        btn_deal.click(
            fn=action_deal,
            inputs=[state, inp_bet],
            outputs=outputs,
        )

        btn_hit.click(fn=action_hit, inputs=[state], outputs=outputs)
        btn_stand.click(fn=action_stand, inputs=[state], outputs=outputs)
        btn_double.click(fn=action_double, inputs=[state], outputs=outputs)
        btn_split.click(fn=action_split, inputs=[state], outputs=outputs)
        btn_surrender.click(fn=action_surrender, inputs=[state], outputs=outputs)
        btn_ins_yes.click(fn=action_insurance_yes, inputs=[state], outputs=outputs)
        btn_ins_no.click(fn=action_insurance_no, inputs=[state], outputs=outputs)

        btn_run_sim.click(
            fn=run_mass_sim,
            inputs=[
                sim_num_hands, sim_num_decks, sim_bet_size,
                sim_counting, sim_min_bet, sim_max_bet,
            ],
            outputs=[sim_results],
        )

    return app, custom_css


# ── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app, custom_css = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=custom_css,
        theme=APP_THEME,
    )
