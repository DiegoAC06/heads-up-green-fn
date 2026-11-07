"""Poker bot controller: M1 equity decisions with M2 opponent adaptation."""

import os

from game import Action
from pokerbot.detect import adjust, classify
from pokerbot.opp_model import OppModel
from pokerbot.strategy import DEFAULT_PARAMS, decide
from pokerbot.timebank import TimeBank


class Bot:
    def __init__(self, params=None, detect_opponents=True):
        # Start from M1 settings; do not mutate the shared defaults.
        self.params = DEFAULT_PARAMS.copy()
        if params is not None:
            self.params.update(params)
        self.detect_opponents = detect_opponents

    def new_half(self, config, memory):
        # Reset everything at the beginning of each 1,000-hand half.
        memory["hands_played"] = 0
        memory["net_chips"] = 0
        memory["internal_errors"] = 0
        memory["last_internal_error"] = None
        memory["starting_stack"] = config.starting_stack
        memory["tb"] = TimeBank(config.time_bank_seconds, 2.0, 0.015)

        # Opponent information never carries between halves.
        memory["opp"] = OppModel()
        memory["label"] = "unknown"
        memory["params"] = self.params.copy()
        memory["last_classified_bucket"] = -1

    def choose_action(self, obs, memory):
        tb = memory["tb"]
        tb.begin()
        try:
            # Read only newly seen opponent actions, avoiding double counts.
            memory["opp"].observe(obs)

            # Re-classify at the first decision in each 25-hand group.
            if self.detect_opponents:
                bucket = obs.hand_index // 25
                if bucket != memory["last_classified_bucket"]:
                    label = classify(memory["opp"])
                    memory["label"] = label
                    # Always apply adjustments to the unmodified M1 settings.
                    memory["params"] = adjust(self.params, label)
                    memory["last_classified_bucket"] = bucket

            return decide(obs, memory, memory["params"])
        finally:
            tb.end()

    def on_hand_end(self, result, memory):
        # An opponent fold can end a hand before their action is observed.
        memory["opp"].hand_end(result)
        memory["hands_played"] += 1
        memory["net_chips"] += result.my_net

    @staticmethod
    def check_or_fold(obs):
        # Preserve the helper used by earlier M0 tests.
        if obs.to_call > 0 and Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL


def make_bot(params=None, detect_opponents=None):
    """Build M2 by default; set POKERBOT_M2=0 to test M1 unchanged."""
    if detect_opponents is None:
        detect_opponents = os.environ.get("POKERBOT_M2", "1") != "0"
    return Bot(params, detect_opponents=detect_opponents)
