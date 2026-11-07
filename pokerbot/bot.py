"""Poker bot controller for the M1 equity-based strategy."""

from game import Action
from pokerbot.strategy import DEFAULT_PARAMS, decide
from pokerbot.timebank import TimeBank


class Bot:
    def __init__(self, params=None):
        # Keep each bot's settings separate from the shared defaults.
        self.params = DEFAULT_PARAMS.copy()
        if params is not None:
            self.params.update(params)

    def new_half(self, config, memory):
        # The tournament resets memory for every half-match.
        memory["hands_played"] = 0
        memory["net_chips"] = 0
        memory["internal_errors"] = 0
        memory["last_internal_error"] = None
        memory["starting_stack"] = config.starting_stack

        # Track the time spent making decisions.
        memory["tb"] = TimeBank(config.time_bank_seconds, 2.0, 0.015)

    def choose_action(self, obs, memory):
        # Measure every decision, even if an exception occurs.
        tb = memory["tb"]
        tb.begin()
        try:
            return decide(obs, memory, self.params)
        finally:
            tb.end()

    def on_hand_end(self, result, memory):
        # Record results without needing the opponent's private cards.
        memory["hands_played"] += 1
        memory["net_chips"] += result.my_net

    @staticmethod
    def check_or_fold(obs):
        # Keep the M0 helper for compatibility.
        if obs.to_call > 0 and Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL


def make_bot(params=None):
    return Bot(params)
