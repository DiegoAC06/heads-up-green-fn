import math

from evaluator import cards_from_str
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
