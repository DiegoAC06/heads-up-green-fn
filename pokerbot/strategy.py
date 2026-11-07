"""M1 poker decisions for Heads-Up Short Deck 5-Card PLO.

Uses Diego's preflop estimates, Monte Carlo equity, and decision time budget.
This module decides actions; bot.py will provide the per-half memory and TimeBank.
"""

import random
import time

from game import Action, Street
from pokerbot.equity import equity

# Diego's current engine uses camelCase. Support the planned snake_case name too.
try:
    from pokerbot.preflop import preflop_pct
except ImportError:
    from pokerbot.preflop import preflopPct as preflop_pct


# Starting values from the M1 strategy plan; tune these after full-match tests.
DEFAULT_PARAMS = {
    "pf_raise_pct": 0.80,
    "pf_call_pct": 0.40,
    "pf_reraise_pct": 0.92,
    "call_margin": 0.03,
    "bet_eq": 0.60,
    "raise_half_eq": 0.62,
    "raise_pot_eq": 0.72,
    "bluff_freq": 0.08,
    "bluff_max_eq": 0.35,
    "range_step": 0.15,
    "range_cap": 0.75,
    "commit_frac": 0.35,
    "commit_eq": 0.40,
    "trap_freq": 0.0,
}


def legal_action(obs, *choices):
    """Return the first legal choice; CHECK_CALL is always legal in this game."""
    for action in choices:
        if action in obs.legal_actions:
            return action
    return Action.CHECK_CALL


def check_or_fold(obs):
    """Never fold if checking is free."""
    if obs.to_call > 0:
        return legal_action(obs, Action.FOLD, Action.CHECK_CALL)
    return Action.CHECK_CALL


def opponent_raises(obs, memory):
    """Count opponent raises this hand, until M2 adds its full model."""
    model = memory.get("opp")
    if model is not None and callable(getattr(model, "raises_this_hand", None)):
        return model.raises_this_hand(obs)
    return sum(
        1 for entry in obs.action_history
        if (not entry.is_me
            and entry.action in (Action.RAISE_HALF, Action.RAISE_POT))
    )


def decide(obs, memory, params=None):
    """Choose a legal action using preflop strength or postflop equity.

    obs: official game.Observation; never reads the opponent's private cards.
    memory: per-half dictionary; bot.py will store a TimeBank as memory['tb'].
    params: adjustable M1 thresholds (M2 can supply overrides later).
    """
    if params is None:
        params = DEFAULT_PARAMS

    # The pot already includes the opponent's wager. Calling adds to_call.
    pot_odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0

    # Only integer values are hashed, keeping decisions reproducible across runs.
    # NumPy requires a non-negative seed; limit the deterministic hash to 32 bits.
    seed = hash((obs.hand_index, int(obs.street), len(obs.action_history))) & 0xFFFFFFFF
    rng = random.Random(seed)

    if obs.street == Street.PREFLOP:
        strength = preflop_pct(obs.hole_cards)
        raised = any(
            not entry.is_me
            and entry.street == Street.PREFLOP
            and entry.action in (Action.RAISE_HALF, Action.RAISE_POT)
            for entry in obs.action_history
        )

        if (strength >= params["pf_reraise_pct"]
                or (strength >= params["pf_raise_pct"] and not raised)):
            return legal_action(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL)

        if (strength >= params["pf_call_pct"]
                or obs.to_call <= 1
                or (pot_odds <= 0.25 and strength >= params["pf_call_pct"] - 0.10)):
            return Action.CHECK_CALL

        return check_or_fold(obs)

    # Postflop, estimate showdown equity against legal opponent hands.
    # The current engine accepts the range threshold but does not yet apply it.
    opp_min_pct = min(
        params["range_cap"],
        params["range_step"] * opponent_raises(obs, memory),
    )
    tb = memory.get("tb")
    deadline = tb.deadline(obs.hand_index) if tb is not None else time.perf_counter() + 0.001

    # Positional arguments work with Diego's current camelCase and planned
    # snake_case parameter spellings; both have the same argument order.
    eq = equity(obs.hole_cards, obs.board, 300, deadline, opp_min_pct, seed)

    # Strong hands: bet/raise for value. All raises are checked for legality.
    if eq >= params["raise_pot_eq"] and rng.random() >= params["trap_freq"]:
        return legal_action(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL)

    if eq >= params["raise_half_eq"] or (obs.to_call == 0 and eq >= params["bet_eq"]):
        return legal_action(obs, Action.RAISE_HALF, Action.CHECK_CALL)

    # A free check cannot be replaced by a fold. Bluff only in position.
    if obs.to_call == 0:
        if (obs.is_button and eq <= params["bluff_max_eq"]
                and rng.random() < params["bluff_freq"]):
            return legal_action(obs, Action.RAISE_HALF, Action.CHECK_CALL)
        return Action.CHECK_CALL

    # A call needs enough estimated equity to cover its price plus a margin.
    if eq >= pot_odds + params["call_margin"]:
        return Action.CHECK_CALL

    # When short-stacked, use the plan's adjusted commitment rule.
    starting_stack = memory.get("starting_stack", 80)
    if ((obs.my_stack - obs.to_call) < params["commit_frac"] * starting_stack
            and eq >= params["commit_eq"]):
        return Action.CHECK_CALL

    return check_or_fold(obs)
