import os
import subprocess
import sys

import numpy as np

from evaluator import cards_from_str
from pokerbot.preflop import PREFLOP_EQ, PREFLOP_PCT, TABLE_LOADED, colexIndex, colexIndexBatch, preflopEquity, preflopPct

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def testTableIsLoadedNotHeuristic():
  assert TABLE_LOADED


def testTableShapeAndRange():
  assert PREFLOP_EQ.shape == (376992,)
  assert PREFLOP_PCT.shape == (376992,)
  # quads are the worst omaha holding (about 0.10), double suited aces the best
  assert 0.05 < PREFLOP_EQ.min() and PREFLOP_EQ.max() < 0.8


def testAverageEquityIsOneHalf():
  # every matchup is counted from both sides, so the mean must be 0.5
  assert abs(float(PREFLOP_EQ.mean()) - 0.5) < 0.005


def testQuadsAreWorseThanDoubleSuitedAces():
  assert preflopPct(cards_from_str("6c 6d 6h 6s Jh")) < 0.01
  assert preflopPct(cards_from_str("Ad As Qd Qs Th")) > 0.99


def testColexEndpoints():
  assert colexIndex([0, 1, 2, 3, 4]) == 0
  assert colexIndex([31, 32, 33, 34, 35]) == 376991


def testColexBatchMatchesScalarAndIgnoresOrder():
  rows = np.array([[4, 3, 2, 1, 0], [35, 31, 33, 32, 34], [7, 20, 3, 11, 30]])
  assert list(colexIndexBatch(rows)) == [0, 376991, colexIndex(sorted([7, 20, 3, 11, 30]))]


def testSuitIsomorphicHandsMatch():
  a = cards_from_str("Ac Ad Kc Kd 9h")
  b = cards_from_str("Ah As Kh Ks 9c")
  assert preflopEquity(a) == preflopEquity(b)
  assert preflopPct(a) == preflopPct(b)


def testStrongBeatsWeak():
  assert preflopPct(cards_from_str("Ac Ad Kc Kd Qh")) > 0.9
  assert preflopPct(cards_from_str("6c 7d 9h Js Kc")) < 0.2


def testLoadsFromOtherCwd(tmp_path):
  code = f"import sys; sys.path.insert(0, {ROOT!r}); import pokerbot.preflop as p; print(p.TABLE_LOADED, p.PREFLOP_EQ.shape)"
  out = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True).stdout
  assert "True (376992,)" in out
