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
    # Tight-aggressive defense: after a preflop raise, assume postflop that the
    # opponent holds at least their inferred raising range. Raising a share p
    # of hands means roughly the top p, i.e. preflop pct >= 1 - p.
    "pf_range_post": True,
    "pf_range_slack": 0.0,     # widen the assumed range by this much
    # Postflop betting range: an opponent who bets or raises a share a of their
    # postflop decisions is assumed, after k such bets this hand, to hold the
    # top (a * post_widen) ** k of hands by current strength.
    "post_range": True,
    "post_widen": 2.0,
    "post_floor": 0.10,
    # Fold-equity bluffing: when checked to with a weak hand, bluff half pot with
    # probability fe_k * (their fold-to-bet on this street - 1/3), capped at 0.6;
    # 1/3 is the break-even fold rate of a half-pot bet. fe_oop allows it out of position.
    "fe_bluff": True,
    "fe_k": 1.5,
    "fe_oop": False,
    # Button opens and steals: raise unopened button hands at or above
    # btn_open_pct; vs opponents who fold to our preflop raises at least
    # steal_fold_min of the time, also raise anything at or above steal_pct.
    "btn_open_pct": 0.65,
    "steal_fold_min": 0.5,
    "steal_pct": 0.30,
    # Adaptive sampling: a pilot run, plus extra samples only when the pilot
    # equity is within close_band of a decision threshold.
    "adaptive_trials": True,
    "pilot_trials": 120,
    "extra_trials": 600,
    "close_band": 0.06,
}

PRIOR_PF_FOLD = 0.35
PRIOR_PF_FOLD_WEIGHT = 15
FE_BREAK_EVEN = 1.0 / 3.0
FE_MAX = 0.6


def raiser_min_pct(memory, params):
    """Weakest preflop pct the opponent is assumed to raise with."""
    model = memory.get("opp")
    pfr = getattr(model, "pfr", None)
    if pfr is None:
        return 0.0
    return max(0.0, min(params["range_cap"], 1.0 - pfr - params.get("pf_range_slack", 0.0)))


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


def post_top_frac(obs, memory, params):
    """Share of opponent hands (by current strength) consistent with their postflop bets."""
    if not params.get("post_range"):
        return 1.0
    model = memory.get("opp")
    rate = getattr(model, "pafOn", None)
    if rate is None:
        return 1.0
    # each postflop bet narrows by that street's bet rate (river-only bluffers stay wide)
    frac = 1.0
    for entry in obs.action_history:
        if (not entry.is_me and entry.street != Street.PREFLOP
                and entry.action in (Action.RAISE_HALF, Action.RAISE_POT)):
            frac *= min(1.0, rate(entry.street) * params.get("post_widen", 1.0))
    return max(params.get("post_floor", 0.10), frac)


def nearest_threshold_gap(eq, obs, params):
    """Distance from eq to the closest equity threshold that decides this spot."""
    pot_odds = obs.to_call / (obs.pot + obs.to_call) if obs.to_call else 0.0
    marks = [params["raise_half_eq"], params["raise_pot_eq"]]
    marks.append(pot_odds + params["call_margin"] if obs.to_call else params["bet_eq"])
    return min(abs(eq - m) for m in marks)


def pf_fold_to_raise(memory):
    """Smoothed share of our preflop raises that the opponent folded to."""
    return ((memory.get("ourPfRaiseFolds", 0) + PRIOR_PF_FOLD * PRIOR_PF_FOLD_WEIGHT)
            / (memory.get("ourPfRaises", 0) + PRIOR_PF_FOLD_WEIGHT))


def _record_pf(obs, memory, action):
    raised = action in (Action.RAISE_HALF, Action.RAISE_POT)
    memory["pfLast"] = (obs.hand_index, raised)
    if raised:
        memory["ourPfRaises"] = memory.get("ourPfRaises", 0) + 1
    return action


def _opp_raised_preflop(obs):
    return any(
        not entry.is_me
        and entry.street == Street.PREFLOP
        and entry.action in (Action.RAISE_HALF, Action.RAISE_POT)
        for entry in obs.action_history
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
            return _record_pf(obs, memory, legal_action(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL))

        # Unopened button: open wider, and steal more vs players who fold to raises.
        if not raised and obs.is_button and (
                strength >= params.get("btn_open_pct", 1.1)
                or (pf_fold_to_raise(memory) >= params.get("steal_fold_min", 1.1)
                    and strength >= params.get("steal_pct", 1.1))):
            return _record_pf(obs, memory, legal_action(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL))

        if (strength >= params["pf_call_pct"]
                or obs.to_call <= 1
                or (pot_odds <= 0.25 and strength >= params["pf_call_pct"] - 0.10)):
            return _record_pf(obs, memory, Action.CHECK_CALL)

        return _record_pf(obs, memory, check_or_fold(obs))

    # Postflop, estimate showdown equity against the opponent's likely range:
    # a preflop-strength floor (raises this hand, inferred raising range) and,
    # after postflop bets, only the top share of hands by current strength.
    opp_min_pct = min(
        params["range_cap"],
        params["range_step"] * opponent_raises(obs, memory),
    )
    if params.get("pf_range_post") and _opp_raised_preflop(obs):
        opp_min_pct = max(opp_min_pct, raiser_min_pct(memory, params))
    tb = memory.get("tb")
    deadline = tb.deadline(obs.hand_index) if tb is not None else time.perf_counter() + 0.001

    # Contract: equity(hole, board, trials, deadline, oppMinPct, seed, oppTopFrac=...).
    top_frac = post_top_frac(obs, memory, params)
    if params.get("adaptive_trials"):
        pilot = params.get("pilot_trials", 120)
        eq = equity(obs.hole_cards, obs.board, pilot, deadline, opp_min_pct, seed, oppTopFrac=top_frac)
        if (nearest_threshold_gap(eq, obs, params) < params.get("close_band", 0.04)
                and time.perf_counter() < deadline):
            extra = params.get("extra_trials", 600)
            eq2 = equity(obs.hole_cards, obs.board, extra, deadline, opp_min_pct,
                         (seed + 1) & 0xFFFFFFFF, oppTopFrac=top_frac)
            eq = (eq * pilot + eq2 * extra) / (pilot + extra)
    else:
        eq = equity(obs.hole_cards, obs.board, 300, deadline, opp_min_pct, seed, oppTopFrac=top_frac)

    # Strong hands: bet/raise for value. All raises are checked for legality.
    if eq >= params["raise_pot_eq"] and rng.random() >= params["trap_freq"]:
        return legal_action(obs, Action.RAISE_POT, Action.RAISE_HALF, Action.CHECK_CALL)

    if eq >= params["raise_half_eq"] or (obs.to_call == 0 and eq >= params["bet_eq"]):
        return legal_action(obs, Action.RAISE_HALF, Action.CHECK_CALL)

    # A free check cannot be replaced by a fold. Bluff only in position.
    if obs.to_call == 0:
        model = memory.get("opp")
        if (params.get("fe_bluff") and eq <= params["bluff_max_eq"]
                and (obs.is_button or params.get("fe_oop"))
                and callable(getattr(model, "foldToBetOn", None))):
            prob = min(FE_MAX, max(0.0, params.get("fe_k", 1.5) * (model.foldToBetOn(obs.street) - FE_BREAK_EVEN)))
            if rng.random() < prob:
                return legal_action(obs, Action.RAISE_HALF, Action.CHECK_CALL)
            return Action.CHECK_CALL
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
