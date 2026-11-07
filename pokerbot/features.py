from itertools import combinations

from evaluator import get_table
from pokerbot.preflop import heuristicStrength


def made_hand(hole, board):
    """Return the best current score using exactly 2 hole + 3 board cards."""
    if len(hole) != 5 or len(board) not in (3, 4, 5):
        raise ValueError("Expected 5 hole cards and 3, 4, or 5 board cards")

    # The evaluator maps each five-card bitmask to a hand score.
    # agent.py already builds this table during startup.
    table = get_table()

    # Represent each card with one bit at its card-number position.
    hole_bits = [1 << card for card in hole]
    board_bits = [1 << card for card in board]

    # Build all legal two-card hole combinations and three-card
    # board combinations. This enforces the Omaha 2 + 3 rule.
    hole_masks = [
        a | b for a, b in combinations(hole_bits, 2)
    ]
    board_masks = [
        a | b | c for a, b, c in combinations(board_bits, 3)
    ]

    # Combine each hole pair with each board triple.
    # Higher evaluator scores represent stronger hands.
    best = 0
    for hole_mask in hole_masks:
        for board_mask in board_masks:
            score = table[hole_mask | board_mask]
            if score > best:
                best = score

    return best


def heuristic_preflop(hole):
    """Return a rough starting-hand rating, not a win probability."""
    if len(hole) != 5:
        raise ValueError("Expected 5 hole cards")

    # Reuse Diego's heuristic until the precomputed table is ready.
    return heuristicStrength(hole)