import os
import subprocess
import sys
import time

from evaluator import cards_from_str
from pokerbot.equity import equity, equityRef
from pokerbot.preflop import preflopEquity, preflopPct
from pokerbot.timebank import TimeBank

AA = cards_from_str("Ac Ad Kh Qs Jh")


def testEquityAllStreetsInRange():
  boards = ([], cards_from_str("6c 7d 8h"), cards_from_str("6c 7d 8h 9s"), cards_from_str("6c 7d 8h 9s Ts"))
  for board in boards:
    e = equity(AA, board, trials=64, seed=1)
    assert 0.0 <= e <= 1.0


def testNutRiverHandIsNearOne():
  # ah kh + qh jh th is a royal flush
  hole = cards_from_str("Ah Kh 6c 7c 8d")
  board = cards_from_str("Qh Jh Th 6s 7s")
  assert equity(hole, board, trials=200, seed=1) == 1.0


def testEquityRefIsSeedReproducible():
  board = cards_from_str("6c 7d 8h")
  assert equityRef(AA, board, trials=100, seed=5) == equityRef(AA, board, trials=100, seed=5)


def testPreflopFunctionsInRange():
  assert 0.0 <= preflopEquity(AA) <= 1.0
  assert 0.0 <= preflopPct(AA) <= 1.0


def testPreflopHeuristicOrdersStrongOverWeak():
  strong = cards_from_str("Ac Ad Kc Kd Qh")
  weak = cards_from_str("6c 7d 9h Js Kc")
  assert preflopPct(strong) > preflopPct(weak)


def testTimebankDeadlineIsBounded():
  tb = TimeBank()
  tb.begin()
  d = tb.deadline(handIndex=0)
  assert 0 <= d - time.perf_counter() <= 0.015
  tb.end()
  assert tb.spent() >= 0


def testTimebankCapsLongBudgets():
  # plenty of bank left on the last hand, so only the 15 ms cap limits it
  tb = TimeBank(totalS=100.0)
  tb.begin()
  assert tb.deadline(handIndex=999) - time.perf_counter() <= 0.015


def testTimebankBudgetHitsZeroWhenSpent():
  tb = TimeBank(totalS=1.0, reserveS=1.0)
  tb.begin()
  assert tb.deadline(handIndex=0) <= time.perf_counter()


def testWarmupIsFast():
  import pokerbot
  t = time.perf_counter()
  pokerbot.warmup()
  assert time.perf_counter() - t < 3.0


def testWarmupMakesFirstDecisionCheap():
  # a fresh process pays table building in warmup, not in the first equity call
  code = ("import time, pokerbot; pokerbot.warmup(); from pokerbot.equity import equity; "
          "t = time.perf_counter(); equity([0, 5, 10, 15, 20], [1, 6, 11], trials=300); "
          "print(time.perf_counter() - t)")
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  out = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
  assert float(out.stdout.strip()) < 0.05, out.stderr
