"""Track an opponent's betting habits during one half-match.

Only reads public actions from Observation.action_history.  A fold that ends a
hand is inferred from the official HandResult, since no further observation
is delivered after the fold.
"""

from game import Action, Street


class OppModel:
    def __init__(self):
        self.decisions = 0
        self.raises = 0
        self.calls = 0
        self.faced = 0
        self.folds = 0
        self._hand_index = None
        self._seen = 0
        self._last_ended = None

    @property
    def raise_rate(self):
        return self.raises / self.decisions if self.decisions else 0.0

    @property
    def call_rate(self):
        return self.calls / self.decisions if self.decisions else 0.0

    @property
    def fold_to_bet(self):
        return self.folds / self.faced if self.faced else 0.0

    def observe(self, obs):
        """Count newly visible opponent actions, never the same action twice."""
        if obs.hand_index != self._hand_index:
            self._hand_index = obs.hand_index
            self._seen = 0

        history = obs.action_history
        if len(history) < self._seen:
            self._seen = 0

        # Reconstruct contributions on each street so CHECK_CALL can be
        # distinguished from a free check. Blinds are absent from history.
        my_bet = obs.big_blind // 2 if obs.is_button else obs.big_blind
        opp_bet = obs.big_blind if obs.is_button else obs.big_blind // 2
        street = Street.PREFLOP

        for i, entry in enumerate(history):
            if entry.street != street:
                street = entry.street
                my_bet = 0
                opp_bet = 0

            if not entry.is_me and i >= self._seen:
                self.decisions += 1
                if my_bet > opp_bet:
                    self.faced += 1
                if entry.action in (Action.RAISE_HALF, Action.RAISE_POT):
                    self.raises += 1
                elif entry.action == Action.CHECK_CALL and my_bet > opp_bet:
                    self.calls += 1

            if entry.is_me:
                my_bet = entry.street_total
            else:
                opp_bet = entry.street_total

        self._seen = len(history)

    def hand_end(self, result):
        """Record an opponent fold, including when opponent cards are hidden."""
        if result.hand_index == self._last_ended:
            return
        self._last_ended = result.hand_index

        if not result.went_to_showdown and result.my_net > 0:
            # Their terminal fold is not included in our next observation.
            self.decisions += 1
            self.faced += 1
            self.folds += 1

        self._hand_index = None
        self._seen = 0

    def raises_this_hand(self, obs):
        """Count opponent raises in the current hand's public history."""
        return sum(
            1 for entry in obs.action_history
            if not entry.is_me and entry.action in (Action.RAISE_HALF, Action.RAISE_POT)
        )
