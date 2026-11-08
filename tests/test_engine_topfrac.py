import time

from evaluator import cards_from_str
from pokerbot.equity import equity

MID = cards_from_str("8c 8d Tc Jd Qh")
FLOP = cards_from_str("6c 7d 9h")


def testTopFracLowersEquityVsABettor():
  # an opponent who only bets with their top 30% of current hands is much stronger
  wide = equity(MID, FLOP, trials=3000, seed=1)
  narrow = equity(MID, FLOP, trials=3000, seed=1, oppTopFrac=0.3)
  assert narrow < wide - 0.05


def testTopFracOneIsUnchanged():
  assert equity(MID, FLOP, trials=500, seed=3) == equity(MID, FLOP, trials=500, seed=3, oppTopFrac=1.0)


def testTopFracIgnoredPreflop():
  assert equity(MID, [], trials=500, seed=3) == equity(MID, [], trials=500, seed=3, oppTopFrac=0.3)


def testTopFracWorksOnTurnAndRiver():
  for board in (cards_from_str("6c 7d 9h Ks"), cards_from_str("6c 7d 9h Ks As")):
    e = equity(MID, board, trials=300, seed=2, oppTopFrac=0.2)
    assert 0.0 <= e <= 1.0


def testNutsStayNearOneVsNarrowRange():
  # a royal flush (Js Ts + As Ks Qs) cannot be beaten, whatever the range
  hole = cards_from_str("Js Ts 9c 8c 6c")
  board = cards_from_str("As Ks Qs 6h 7d")
  assert equity(hole, board, trials=300, seed=1, oppTopFrac=0.1) > 0.95


def testTopFracWithPreflopRangeIsFast():
  hole = cards_from_str("Ac Ad Kc Kd Th")
  equity(hole, FLOP, trials=64, seed=0, oppMinPct=0.6, oppTopFrac=0.3)  # warm numpy paths
  times = []
  for s in range(30):
    t = time.perf_counter()
    equity(hole, FLOP, trials=300, seed=s, oppMinPct=0.6, oppTopFrac=0.3)
    times.append(time.perf_counter() - t)
  times.sort()
  # sanity bound only: in a long pytest run timing is noisy; tools/bench_engine.py measures it (about 1.8 ms)
  assert times[len(times) // 2] < 0.015
