"""Preflop hand strength from a precomputed table of all 376,992 hands.

data/preflop_eq.npy holds each hand's equity against a random hand, indexed by
the colex index of its sorted cards (built by tools/build_preflop.py). If the
file or numpy is missing, a heuristic keeps the bot running.
"""

import os
from collections import Counter
from math import comb
from typing import Sequence

from evaluator import card_rank, card_suit

NUM_HANDS = comb(36, 5)  # 376,992
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "preflop_eq.npy")

# COMB[c][k] = c choose k, for the colex index
COMB = [[comb(c, k) for k in range(6)] for c in range(36)]


def colexIndex(cards: Sequence[int]) -> int:
  """Unique 0..376,991 index of a 5-card set (any order)."""
  return sum(COMB[c][i + 1] for i, c in enumerate(sorted(cards)))


def heuristicStrength(hole: Sequence[int]) -> float:
  """Rough 0..1 score from pairs, suits, connectedness and high cards."""
  ranks = [card_rank(c) for c in hole]
  rankCounts = Counter(ranks)
  suitCounts = Counter(card_suit(c) for c in hole)

  pairs = sum(1 for n in rankCounts.values() if n == 2)
  trips = sum(1 for n in rankCounts.values() if n >= 3)
  suited = sum(1 for n in suitCounts.values() if n >= 2)

  distinct = sorted(rankCounts)
  connected = sum(1 for a, b in zip(distinct, distinct[1:]) if b - a <= 2)
  highCards = sum(r for r in ranks if r >= 5) / (5 * 8)  # j and up

  score = (0.18 * pairs + 0.05 * trips + 0.12 * suited
           + 0.06 * connected + 0.35 * highCards)
  return max(0.0, min(1.0, score))


def _load():
  try:
    import numpy as np
    eq = np.load(DATA_PATH).astype(np.float32)
    if eq.shape != (NUM_HANDS,):
      return None, None
  except Exception:
    return None, None
  # mean rank for ties, so suit-isomorphic hands get the same percentile
  ordered = np.sort(eq)
  lo = np.searchsorted(ordered, eq, side="left")
  hi = np.searchsorted(ordered, eq, side="right") - 1
  pct = ((lo + hi) / 2.0 / (NUM_HANDS - 1)).astype(np.float32)
  return eq, pct


PREFLOP_EQ, PREFLOP_PCT = _load()
TABLE_LOADED = PREFLOP_EQ is not None

if TABLE_LOADED:
  import numpy as np
  COMB_NP = np.array(COMB, dtype=np.int64)
  KS = np.arange(1, 6)

  def colexIndexBatch(cards):
    """(N,5) cards in any order -> (N,) colex indices."""
    rows = np.sort(np.asarray(cards, dtype=np.int64), axis=1)
    return COMB_NP[rows, KS].sum(axis=1)
else:
  def colexIndexBatch(cards):
    return [colexIndex(row) for row in cards]


def preflopEquity(hole: Sequence[int]) -> float:
  """Equity against a random hand, 0..1."""
  if TABLE_LOADED:
    return float(PREFLOP_EQ[colexIndex(hole)])
  return 0.3 + 0.4 * heuristicStrength(hole)


def preflopPct(hole: Sequence[int]) -> float:
  """Percentile among all hands, 0 worst .. 1 best."""
  if TABLE_LOADED:
    return float(PREFLOP_PCT[colexIndex(hole)])
  return heuristicStrength(hole)
