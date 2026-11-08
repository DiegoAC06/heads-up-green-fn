"""Track an opponent's betting habits during one half-match.

Only reads public actions from Observation.action_history.  A fold that ends a
hand is inferred from the official HandResult, since no further observation
is delivered after the fold.
"""

from game import Action, Street

# Preflop raise rate starts at this prior and moves toward observed counts.
PRIOR_PFR = 0.25
PRIOR_PFR_WEIGHT = 20
# Postflop bet/raise rate per postflop decision, same smoothing idea.
PRIOR_PAF = 0.30
PRIOR_PAF_WEIGHT = 20
# Per-street rates shrink toward the overall postflop rate with this weight.
STREET_PAF_WEIGHT = 15
POSTFLOP_STREETS = (Street.FLOP, Street.TURN, Street.RIVER)
# Fold-to-bet per street, shrunk toward their overall fold-to-bet.
PRIOR_FOLD = 0.35
STREET_FOLD_WEIGHT = 15


class OppModel:
    def __init__(self):
        # Hands where the opponent acted preflop, and how many of those they raised.
        self.pfOpps = 0
        self.pfRaises = 0
        self._pfActed = False
        self._pfRaised = False
        # Opponent postflop decisions and how many were bets or raises.
        self.postDecisions = 0
        self.postRaises = 0
        self.streetDecisions = {st: 0 for st in POSTFLOP_STREETS}
        self.streetRaises = {st: 0 for st in POSTFLOP_STREETS}
        self.streetFaced = {st: 0 for st in POSTFLOP_STREETS}
        self.streetFolds = {st: 0 for st in POSTFLOP_STREETS}
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
    def pfr(self):
        """Smoothed share of hands in which the opponent raised preflop."""
        return ((self.pfRaises + PRIOR_PFR * PRIOR_PFR_WEIGHT)
                / (self.pfOpps + PRIOR_PFR_WEIGHT))

    @property
    def paf(self):
        """Smoothed share of postflop decisions that were bets or raises."""
        return ((self.postRaises + PRIOR_PAF * PRIOR_PAF_WEIGHT)
                / (self.postDecisions + PRIOR_PAF_WEIGHT))

    def pafOn(self, street):
        """Bet/raise rate on one postflop street, shrunk toward the overall rate."""
        overall = self.paf
        return ((self.streetRaises.get(street, 0) + overall * STREET_PAF_WEIGHT)
                / (self.streetDecisions.get(street, 0) + STREET_PAF_WEIGHT))

    def foldToBetOn(self, street):
        """Fold-to-bet on one postflop street, shrunk toward the overall rate."""
        overall = self.folds / self.faced if self.faced else PRIOR_FOLD
        return ((self.streetFolds.get(street, 0) + overall * STREET_FOLD_WEIGHT)
                / (self.streetFaced.get(street, 0) + STREET_FOLD_WEIGHT))

    @property
    def fold_to_bet(self):
        return self.folds / self.faced if self.faced else 0.0

    def observe(self, obs):
        """Count newly visible opponent actions, never the same action twice."""
        if obs.hand_index != self._hand_index:
            self._hand_index = obs.hand_index
            self._seen = 0
            self._pfActed = False
            self._pfRaised = False

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
                    if entry.street != Street.PREFLOP:
                        self.streetFaced[entry.street] += 1
                if entry.action in (Action.RAISE_HALF, Action.RAISE_POT):
                    self.raises += 1
                elif entry.action == Action.CHECK_CALL and my_bet > opp_bet:
                    self.calls += 1

                if entry.street != Street.PREFLOP:
                    self.postDecisions += 1
                    self.streetDecisions[entry.street] += 1
                    if entry.action in (Action.RAISE_HALF, Action.RAISE_POT):
                        self.postRaises += 1
                        self.streetRaises[entry.street] += 1

                # One preflop opportunity per hand; one raise at most per hand.
                if entry.street == Street.PREFLOP:
                    if not self._pfActed:
                        self._pfActed = True
                        self.pfOpps += 1
                    if (entry.action in (Action.RAISE_HALF, Action.RAISE_POT)
                            and not self._pfRaised):
                        self._pfRaised = True
                        self.pfRaises += 1

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
            # Folding before any preflop action still used a preflop opportunity.
            if not self._pfActed:
                self.pfOpps += 1
            # A fold with board cards out was a postflop decision.
            if len(result.board) >= 3:
                self.postDecisions += 1
                last = {3: Street.FLOP, 4: Street.TURN, 5: Street.RIVER}[min(len(result.board), 5)]
                self.streetDecisions[last] += 1
                self.streetFaced[last] += 1
                self.streetFolds[last] += 1

        self._hand_index = None
        self._seen = 0
        self._pfActed = False
        self._pfRaised = False

    def raises_this_hand(self, obs):
        """Count opponent raises in the current hand's public history."""
        return sum(
            1 for entry in obs.action_history
            if not entry.is_me and entry.action in (Action.RAISE_HALF, Action.RAISE_POT)
        )
