"""Monte Carlo hand equity for short deck plo5.

equity() is the public entry point. equityRef() is the pure python reference,
kept as the fallback and as the yardstick for faster versions.
"""

import random
import time
from typing import Optional, Sequence

from evaluator import NUM_CARDS, omaha_score

MIN_TRIALS = 8


def equityRef(hole: Sequence[int], board: Sequence[int], trials: int = 300,
              deadline: Optional[float] = None, seed: Optional[int] = None) -> float:
  """P(win) + 0.5 * P(tie) against one uniformly random opponent hand."""
  rng = random.Random(seed)
  used = set(hole) | set(board)
  deck = [c for c in range(NUM_CARDS) if c not in used]
  missing = 5 - len(board)
  board = list(board)
  hole = list(hole)
  total = 0.0
  done = 0
  while done < trials:
    # always run a few trials so a spent deadline still gives a usable number
    if done >= MIN_TRIALS and deadline is not None and time.perf_counter() >= deadline:
      break
    draw = rng.sample(deck, missing + 5)
    fullBoard = board + draw[:missing]
    mine = omaha_score(hole, fullBoard)
    theirs = omaha_score(draw[missing:], fullBoard)
    if mine > theirs:
      total += 1.0
    elif mine == theirs:
      total += 0.5
    done += 1
  return total / done


def equity(hole: Sequence[int], board: Sequence[int], trials: int = 300,
           deadline: Optional[float] = None, oppMinPct: float = 0.0,
           seed: Optional[int] = None) -> float:
  """Contract entry point. oppMinPct is accepted but ignored until task 5."""
  return equityRef(hole, board, trials=trials, deadline=deadline, seed=seed)
