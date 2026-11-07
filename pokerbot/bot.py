from game import Action, Street
from evaluator import THREE_OF_A_KIND, TWO_PAIR
from pokerbot.features import made_hand, heuristic_preflop


class Bot:
    def __init__(self, params=None):
        # These are starting thresholds for M0, not proven optimal values.
        # Copy overrides so we do not change the caller's dictionary.
        self.params = {
            "pf_raise": 0.75,
            "pf_call": 0.40,
            "cheap_call": 1,
            "postflop_call_fraction": 0.25,
        }
        if params is not None:
            self.params.update(params)

    def new_half(self, config, memory):
        # Reset all state owned by this baseline at the start of each half.
        memory["hands_played"] = 0
        memory["net_chips"] = 0
        memory["internal_errors"] = 0
        memory["last_internal_error"] = None

    def choose_action(self, obs, memory):
        if obs.street == Street.PREFLOP:
            strength = heuristic_preflop(obs.hole_cards)

            # Raise strong starting hands when the game permits it.
            if strength >= self.params["pf_raise"]:
                if Action.RAISE_HALF in obs.legal_actions:
                    return Action.RAISE_HALF
                return Action.CHECK_CALL

            # Stay in with reasonable starting hands or a cheap call.
            if (
                strength >= self.params["pf_call"]
                or obs.to_call <= self.params["cheap_call"]
            ):
                return Action.CHECK_CALL

            # Weak hands can still check when no chips are owed.
            return self.check_or_fold(obs)

        # Evaluate only the cards currently visible on the board.
        # The upper bits of the score contain the hand category.
        score = made_hand(obs.hole_cards, obs.board)
        category = score >> 20

        # M0 bets trips or better aggressively. This simple rule will
        # later be replaced by equity and opponent-aware decisions.
        if category >= THREE_OF_A_KIND:
            if Action.RAISE_POT in obs.legal_actions:
                return Action.RAISE_POT
            if Action.RAISE_HALF in obs.legal_actions:
                return Action.RAISE_HALF
            return Action.CHECK_CALL

        # Continue with two pair or better, or when calling is cheap
        # relative to the current pot. This is a heuristic, not pot odds.
        cheap_call = (
            obs.to_call
            <= obs.pot * self.params["postflop_call_fraction"]
        )

        if category >= TWO_PAIR or cheap_call:
            return Action.CHECK_CALL

        return self.check_or_fold(obs)

    def on_hand_end(self, result, memory):
        # Record results without requiring opponent cards.
        # Folded hands have opp_hole=None, and split pots are valid.
        memory["hands_played"] += 1
        memory["net_chips"] += result.my_net

    @staticmethod
    def check_or_fold(obs):
        # Folding is only allowed when we owe chips.
        if obs.to_call > 0 and Action.FOLD in obs.legal_actions:
            return Action.FOLD
        return Action.CHECK_CALL


def make_bot(params=None):
    return Bot(params)