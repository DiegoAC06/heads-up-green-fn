"""Monte Carlo hand equity for short deck plo5.

equity() is the public entry point. equityRef() is the pure python reference,
kept as the fallback and as the yardstick for faster versions.
"""

import math
import random
import time
from itertools import combinations, combinations_with_replacement
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


# ---------------------------------------------------------------------------
# fast path: vectorized numpy scoring
#
# a 5-card hand's score depends only on its rank multiset and whether it is a
# flush. rank key = sum of 5**rank and suit key = sum of 6**suit are both
# additive, so scoring a hole pair plus a board triple is two additions and a
# table lookup, with no sorting.
# ---------------------------------------------------------------------------

try:
  import numpy as np
  HAVE_NUMPY = True
except ImportError:
  np = None
  HAVE_NUMPY = False

BATCH = 64
RANGE_ROUNDS = 2  # draw rounds per batch when filtering by opponent range
PAIRS = tuple(combinations(range(5), 2))
TRIPLES = tuple(combinations(range(5), 3))


def _buildTables():
  from evaluator import score_five
  nonFlush = np.zeros(5 ** 9, dtype=np.int32)
  flush = np.zeros(5 ** 9, dtype=np.int32)
  for ranks in combinations_with_replacement(range(9), 5):
    if max(ranks.count(r) for r in ranks) > 4:
      continue
    key = sum(5 ** r for r in ranks)
    # give repeated ranks different suits; break any accidental flush
    seen: dict = {}
    cards = []
    for r in ranks:
      cards.append(r * 4 + seen.get(r, 0))
      seen[r] = seen.get(r, 0) + 1
    if len(set(ranks)) == 5:
      cards[0] = ranks[0] * 4 + 1
      flush[key] = score_five([r * 4 for r in ranks])
    nonFlush[key] = score_five(cards)
  isFlush = np.zeros(6 ** 4, dtype=bool)
  for s in range(4):
    isFlush[5 * 6 ** s] = True
  rankKey = np.array([5 ** (c >> 2) for c in range(NUM_CARDS)], dtype=np.int64)
  suitKey = np.array([6 ** (c & 3) for c in range(NUM_CARDS)], dtype=np.int64)
  return nonFlush, flush, isFlush, rankKey, suitKey


if HAVE_NUMPY:
  from pokerbot.preflop import PREFLOP_PCT, TABLE_LOADED, colexIndexBatch
  NON_FLUSH, FLUSH_TABLE, IS_FLUSH, RANK_KEY, SUIT_KEY = _buildTables()
  PAIR_A = np.array([p[0] for p in PAIRS])
  PAIR_B = np.array([p[1] for p in PAIRS])
  TRI_A = np.array([t[0] for t in TRIPLES])
  TRI_B = np.array([t[1] for t in TRIPLES])
  TRI_C = np.array([t[2] for t in TRIPLES])


def _pairKeys(cards):
  rk, sk = RANK_KEY[cards], SUIT_KEY[cards]
  return rk[..., PAIR_A] + rk[..., PAIR_B], sk[..., PAIR_A] + sk[..., PAIR_B]


def _tripleKeys(cards):
  rk, sk = RANK_KEY[cards], SUIT_KEY[cards]
  return (rk[..., TRI_A] + rk[..., TRI_B] + rk[..., TRI_C],
          sk[..., TRI_A] + sk[..., TRI_B] + sk[..., TRI_C])


def _bestScores(pairRk, pairSk, triRk, triSk):
  """pair keys (N,10) or (10,), triple keys (N,10) -> best score per row (N,)."""
  rk = pairRk[..., :, None] + triRk[:, None, :]
  sk = pairSk[..., :, None] + triSk[:, None, :]
  return np.where(IS_FLUSH[sk], FLUSH_TABLE[rk], NON_FLUSH[rk]).max(axis=(1, 2))


def omahaScoreBatch(holes, boards):
  """(N,5) holes and (N,5) boards -> (N,) scores, equal to omaha_score per row."""
  holes = np.asarray(holes, dtype=np.int64)
  boards = np.asarray(boards, dtype=np.int64)
  pairRk, pairSk = _pairKeys(holes)
  triRk, triSk = _tripleKeys(boards)
  return _bestScores(pairRk, pairSk, triRk, triSk)


def _drawRows(rng, deck, size: int, need: int, missing: int, oppMinPct: float):
  """size rows of `need` cards; opponent hands (columns missing:) filtered by preflop pct."""
  if oppMinPct <= 0.0 or not TABLE_LOADED:
    return deck[rng.random((size, len(deck))).argsort(axis=1)[:, :need]]
  # oversample with a 1.5x margin (blockers shrink the in-range share), up to two
  # rounds; if still short, top up with out-of-range rows so a batch never starves
  draws = min(math.ceil(1.5 * size / max(1.0 - oppMinPct, 0.25)), 4 * size)
  kept = []
  rest = None
  have = 0
  for _ in range(RANGE_ROUNDS):
    rows = deck[rng.random((draws, len(deck))).argsort(axis=1)[:, :need]]
    inRange = PREFLOP_PCT[colexIndexBatch(rows[:, missing:])] >= oppMinPct
    kept.append(rows[inRange])
    have += int(inRange.sum())
    if rest is None:
      rest = rows[~inRange]
    if have >= size:
      break
  return np.concatenate(kept + [rest])[:size]


def equityFast(hole: Sequence[int], board: Sequence[int], trials: int = 300,
               deadline: Optional[float] = None, oppMinPct: float = 0.0,
               seed: Optional[int] = None) -> float:
  rng = np.random.default_rng(seed)
  used = set(hole) | set(board)
  deck = np.array([c for c in range(NUM_CARDS) if c not in used], dtype=np.int64)
  missing = 5 - len(board)
  need = missing + 5
  boardArr = np.array(list(board), dtype=np.int64)
  myRk, mySk = _pairKeys(np.array(list(hole), dtype=np.int64))
  total = 0.0
  done = 0
  while done < trials:
    if done > 0 and deadline is not None and time.perf_counter() >= deadline:
      break
    size = min(BATCH, trials - done)
    # first `need` columns of a random permutation per row = a draw without replacement
    drawn = _drawRows(rng, deck, size, need, missing, oppMinPct)
    fullBoard = np.concatenate([np.broadcast_to(boardArr, (size, len(boardArr))), drawn[:, :missing]], axis=1)
    triRk, triSk = _tripleKeys(fullBoard)
    mine = _bestScores(myRk, mySk, triRk, triSk)
    oppRk, oppSk = _pairKeys(drawn[:, missing:])
    theirs = _bestScores(oppRk, oppSk, triRk, triSk)
    total += float((mine > theirs).sum()) + 0.5 * float((mine == theirs).sum())
    done += size
  return total / done


def equity(hole: Sequence[int], board: Sequence[int], trials: int = 300,
           deadline: Optional[float] = None, oppMinPct: float = 0.0,
           seed: Optional[int] = None) -> float:
  """Contract entry point. oppMinPct in 0..1 restricts the opponent to hands at
  or above that preflop percentile (numpy path only)."""
  if HAVE_NUMPY:
    return equityFast(hole, board, trials=trials, deadline=deadline, oppMinPct=oppMinPct, seed=seed)
  return equityRef(hole, board, trials=trials, deadline=deadline, seed=seed)
