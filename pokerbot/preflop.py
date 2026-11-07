"""Preflop hand strength.

For now these are heuristics. Task 4 replaces them with a precomputed table.
"""

from collections import Counter
from typing import Sequence

from evaluator import card_rank, card_suit


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


def preflopEquity(hole: Sequence[int]) -> float:
  """Equity against a random hand, 0..1 (heuristic until the table lands)."""
  return 0.3 + 0.4 * heuristicStrength(hole)


def preflopPct(hole: Sequence[int]) -> float:
  """Percentile among all hands, 0 worst .. 1 best (heuristic for now)."""
  return heuristicStrength(hole)
