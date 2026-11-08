"""Per decision time budgeting against the 8 second per half time bank."""

import time

OVERHEAD_PER_DECISION_S = 0.0003  # request handling the judge charges but we cannot see
STARVE_MARGIN_S = 1.0             # this close to the real bank, equity uses a tiny batch
STARVED_DEADLINE_PAST_S = 1.0e6   # deadline this far in the past is equity's starve signal


class TimeBank:
  def __init__(self, totalS: float = 8.0, reserveS: float = 2.0, maxPerDecisionS: float = 0.015) -> None:
    self.totalS = totalS
    self.reserveS = reserveS
    self.maxPerDecisionS = maxPerDecisionS
    self.usedS = 0.0
    self.decisions = 0
    self.t0 = time.perf_counter()

  def begin(self) -> None:
    """Call first thing in choose_action."""
    self.t0 = time.perf_counter()

  def end(self) -> None:
    """Call just before choose_action returns."""
    self.usedS += time.perf_counter() - self.t0
    self.decisions += 1

  def spent(self) -> float:
    return self.usedS + self.decisions * OVERHEAD_PER_DECISION_S

  def deadline(self, handIndex: int) -> float:
    """Absolute perf_counter time this decision should finish by."""
    if self.spent() >= self.totalS - STARVE_MARGIN_S:
      return self.t0 - STARVED_DEADLINE_PAST_S
    remaining = self.totalS - self.reserveS - self.spent()
    estLeft = max(1.0, (1000 - handIndex) * 2.5)
    budget = min(max(remaining / estLeft, 0.0), self.maxPerDecisionS)
    return self.t0 + budget
