"""Time the equity engine per street.

Run from the repo root:
  python tools/bench_engine.py
Target: flop median <= 2 ms at 300 trials, import < 3 s.
"""

import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

t0 = time.perf_counter()
from pokerbot.equity import HAVE_NUMPY, equity  # noqa: E402
importS = time.perf_counter() - t0

CALLS = 200
TRIALS = 300


def benchStreet(boardSize: int, rng: random.Random, oppMinPct: float = 0.0, oppTopFrac: float = 1.0) -> list:
  times = []
  for i in range(CALLS):
    cards = rng.sample(range(36), 5 + boardSize)
    start = time.perf_counter()
    equity(cards[:5], cards[5:], trials=TRIALS, seed=i, oppMinPct=oppMinPct, oppTopFrac=oppTopFrac)
    times.append((time.perf_counter() - start) * 1000)
  return times


def main() -> None:
  print(f"numpy fast path: {HAVE_NUMPY}")
  print(f"import pokerbot.equity: {importS:.2f} s")
  rng = random.Random(0)
  equity(list(range(5)), [], trials=64, seed=0)  # warm caches
  print(f"{'street':<10}{'median ms':>12}{'p95 ms':>10}")
  for name, size in (("preflop", 0), ("flop", 3), ("turn", 4), ("river", 5)):
    times = sorted(benchStreet(size, rng))
    p95 = times[int(len(times) * 0.95) - 1]
    print(f"{name:<10}{statistics.median(times):>12.2f}{p95:>10.2f}")
  # opponent range filtering costs extra draws; target flop median <= 3 ms
  times = sorted(benchStreet(3, rng, oppMinPct=0.6))
  p95 = times[int(len(times) * 0.95) - 1]
  print(f"{'flop@0.6':<10}{statistics.median(times):>12.2f}{p95:>10.2f}")
  # opponent betting-range filter (top 30% of current hands); target flop median <= 3 ms
  times = sorted(benchStreet(3, rng, oppMinPct=0.6, oppTopFrac=0.3))
  p95 = times[int(len(times) * 0.95) - 1]
  print(f"{'flop@top30':<10}{statistics.median(times):>12.2f}{p95:>10.2f}")


if __name__ == "__main__":
  main()
