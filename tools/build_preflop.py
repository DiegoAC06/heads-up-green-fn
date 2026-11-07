"""Build data/preflop_eq.npy: equity vs a random hand for all 376,992 hole hands.

Hands that differ only by a suit relabeling have the same equity, so each
suit-isomorphism class is simulated once and copied to all its members.

Run from the repo root (takes a few minutes):
  python tools/build_preflop.py [--trials 2000]
"""

import argparse
import os
import sys
import time
from itertools import combinations, permutations
from multiprocessing import Pool

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from pokerbot.preflop import DATA_PATH, NUM_HANDS, colexIndex  # noqa: E402

SUIT_PERMS = tuple(permutations(range(4)))


def canonical(hand: tuple) -> tuple:
  """Smallest sorted relabeling of the hand over all 24 suit permutations."""
  return min(tuple(sorted((c & ~3) | p[c & 3] for c in hand)) for p in SUIT_PERMS)


def classEquity(job: tuple) -> float:
  classId, rep, trials = job
  from pokerbot.equity import equity
  return equity(list(rep), [], trials=trials, seed=classId)


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--trials", type=int, default=2000)
  args = p.parse_args()

  start = time.perf_counter()
  classOf = np.empty(NUM_HANDS, dtype=np.int32)
  classIds: dict = {}
  reps: list = []
  for hand in combinations(range(36), 5):
    key = canonical(hand)
    cid = classIds.get(key)
    if cid is None:
      cid = len(reps)
      classIds[key] = cid
      reps.append(key)
    classOf[colexIndex(hand)] = cid
  print(f"{len(reps)} suit classes found in {time.perf_counter() - start:.1f} s")

  jobs = [(cid, rep, args.trials) for cid, rep in enumerate(reps)]
  with Pool(os.cpu_count() or 1) as pool:
    classEq = np.array(pool.map(classEquity, jobs, chunksize=200), dtype=np.float32)
  print(f"simulated {len(reps)} classes x {args.trials} trials in {time.perf_counter() - start:.1f} s")

  table = classEq[classOf]
  os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
  np.save(DATA_PATH, table)
  print(f"wrote {DATA_PATH}: min {table.min():.3f} max {table.max():.3f} mean {table.mean():.3f}")


if __name__ == "__main__":
  main()
