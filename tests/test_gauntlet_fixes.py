"""Regression tests for the changes accepted by the Nightmare Gauntlet lab."""

from pokerbot.detect import adjust
from pokerbot.strategy import DEFAULT_PARAMS


def testNitLabelStealsAndRespectsRaises():
  # vs a nit (folds most hands to a bet, raises only strong ones): steal more, call raises less
  p = adjust(DEFAULT_PARAMS, "nit")
  assert p["pf_raise_pct"] == 0.40 and p["bluff_freq"] == 0.5 and p["bluff_max_eq"] == 0.5
  assert p["pf_call_pct"] == 0.75 and p["range_step"] == 0.6 and p["call_margin"] == 0.10


def testOtherLabelsUnchanged():
  for label in ("unknown", "solid", "maniac", "random"):
    assert adjust(DEFAULT_PARAMS, label) == DEFAULT_PARAMS



def testSpentDeadlineRunsOnlyASmallBatch():
  # when the bank signals the hard margin (deadline far in the past), use STARVED_TRIALS
  import time
  from evaluator import cards_from_str
  from pokerbot.equity import STARVED_TRIALS, equity
  assert STARVED_TRIALS == 16
  hole, board = cards_from_str("8c 8d Tc Jd Qh"), cards_from_str("6c 7d 9h")
  for seed in range(8):
    e = equity(hole, board, trials=300, deadline=time.perf_counter() - 1e6, seed=seed)
    assert abs(e * 2 * STARVED_TRIALS - round(e * 2 * STARVED_TRIALS)) < 1e-9


def testSoftSpentDeadlineStillRunsAFullBatch():
  # a deadline that just passed (soft budget spent) behaves like the original: one 64 batch
  import time
  from evaluator import cards_from_str
  from pokerbot.equity import equity
  hole, board = cards_from_str("8c 8d Tc Jd Qh"), cards_from_str("6c 7d 9h")
  vals = [equity(hole, board, trials=300, deadline=time.perf_counter() - 0.001, seed=s) for s in range(8)]
  assert any(abs(v * 32 - round(v * 32)) > 1e-9 for v in vals)


def testLiveDeadlineStillRunsFullTrials():
  import time
  from evaluator import cards_from_str
  from pokerbot.equity import equity
  hole, board = cards_from_str("8c 8d Tc Jd Qh"), cards_from_str("6c 7d 9h")
  vals = [equity(hole, board, trials=300, deadline=time.perf_counter() + 1.0, seed=s) for s in range(8)]
  assert any(abs(v * 32 - round(v * 32)) > 1e-9 for v in vals)


def testBankKeepsFullBatchUntilHardMargin():
  # past the soft budget (totalS - reserveS) but before the hard margin, the deadline is
  # only just passed, so equity runs one full batch exactly like the original bot
  import time
  from pokerbot.timebank import STARVE_MARGIN_S, TimeBank
  tb = TimeBank(8.0, 2.0, 0.015)
  tb.usedS = 6.5
  tb.begin()
  assert tb.deadline(500) > time.perf_counter() - 1.0      # just passed, not the starve signal
  tb.usedS = 8.0 - STARVE_MARGIN_S
  tb.begin()
  assert tb.deadline(500) < time.perf_counter() - 1000.0   # starve signal
