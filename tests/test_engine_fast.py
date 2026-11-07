import importlib
import random
import sys
import time

import numpy as np

from evaluator import cards_from_str, omaha_score
from pokerbot.equity import equity, equityRef, omahaScoreBatch

AA = cards_from_str("Ac Ad Kh Qs Jh")


def testBatchScoreMatchesReferenceExactly():
  rng = random.Random(0)
  holes, boards = [], []
  for _ in range(3000):
    c = rng.sample(range(36), 10)
    holes.append(c[:5])
    boards.append(c[5:])
  got = omahaScoreBatch(np.array(holes), np.array(boards))
  assert [int(x) for x in got] == [omaha_score(h, b) for h, b in zip(holes, boards)]


def testBatchScoreHandlesShortDeckSpecials():
  # wheel a-6-7-8-9, flush over full house, trips over straight
  hole = cards_from_str("As 6d Kc Kd Qh")
  board = cards_from_str("7c 8h 9s Tc Jd")
  assert int(omahaScoreBatch(np.array([hole]), np.array([board]))[0]) == omaha_score(hole, board)


def testFastEquityAgreesWithReference():
  rng = random.Random(1)
  for _ in range(10):
    c = rng.sample(range(36), 8)
    fast = equity(c[:5], c[5:], trials=4000, seed=2)
    ref = equityRef(c[:5], c[5:], trials=4000, seed=3)
    assert abs(fast - ref) < 0.03


def testFastEquityAllStreets():
  for board in ([], cards_from_str("6c 7d 8h"), cards_from_str("6c 7d 8h 9s"), cards_from_str("6c 7d 8h 9s Ts")):
    assert 0.0 <= equity(AA, board, trials=200, seed=4) <= 1.0


def testFastEquityIsSeedReproducible():
  board = cards_from_str("6c 7d 8h")
  assert equity(AA, board, trials=300, seed=7) == equity(AA, board, trials=300, seed=7)


def testExpiredDeadlineStillReturns():
  e = equity(AA, cards_from_str("6c 7d 8h"), trials=300, deadline=time.perf_counter() - 1)
  assert 0.0 <= e <= 1.0


def testFallsBackWithoutNumpy(monkeypatch):
  # a None entry makes "import numpy" raise ImportError
  monkeypatch.setitem(sys.modules, "numpy", None)
  monkeypatch.delitem(sys.modules, "pokerbot.equity", raising=False)
  mod = importlib.import_module("pokerbot.equity")
  assert not mod.HAVE_NUMPY
  assert 0.0 <= mod.equity(AA, [], trials=16, seed=1) <= 1.0
