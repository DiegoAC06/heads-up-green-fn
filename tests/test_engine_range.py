import math

import numpy as np

from evaluator import cards_from_str
import pokerbot.equity as E
from pokerbot.equity import equity

MID = cards_from_str("8c 8d Tc Jd Qh")


def testTighterRangeLowersEquity():
  assert equity(MID, [], trials=3000, seed=1, oppMinPct=0.8) < equity(MID, [], trials=3000, seed=1) - 0.02


def testExtremeRangeDoesNotStarve():
  e = equity(MID, cards_from_str("6c 7d 8h"), trials=300, seed=1, oppMinPct=0.995)
  assert 0.0 <= e <= 1.0 and not math.isnan(e)


def testZeroRangeMatchesUnfiltered():
  # oppMinPct=0 must not change the sampling at all
  board = cards_from_str("6c 7d 8h")
  assert equity(MID, board, trials=500, seed=3, oppMinPct=0.0) == equity(MID, board, trials=500, seed=3)


def testDeadlineStillRespectedWithRange():
  import time
  start = time.perf_counter()
  e = equity(MID, [], trials=10 ** 6, deadline=start + 0.02, seed=1, oppMinPct=0.6)
  assert time.perf_counter() - start < 0.5 and 0.0 <= e <= 1.0


def testRangeFilterHoldsAtStrategyRanges():
  # strategy uses oppMinPct up to 0.75; holding blockers (aces and kings) shrinks the
  # in-range share below 1-p, so oversampling by exactly 1/(1-p) leaks out-of-range hands
  hole = cards_from_str("Ac Ad Kc Kd Th")
  board = cards_from_str("6h 7s 8d")
  deck = np.array([c for c in range(36) if c not in hole + board], dtype=np.int64)
  rng = np.random.default_rng(5)
  for p in (0.3, 0.6, 0.75):
    rows = np.concatenate([E._drawRows(rng, deck, 64, 7, 2, p) for _ in range(100)])
    share = float((E.PREFLOP_PCT[E.colexIndexBatch(rows[:, 2:])] >= p).mean())
    assert share >= 0.995, (p, share)
