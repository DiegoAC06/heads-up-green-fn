def warmup() -> None:
  """Load the evaluator tables and preflop table and run one tiny equity call,
  so the first real decision does not pay for setup. Call it at server start."""
  from pokerbot.equity import equity
  from pokerbot.preflop import preflopPct
  preflopPct([0, 5, 10, 15, 20])
  equity([0, 5, 10, 15, 20], [1, 6, 11], trials=64, seed=0, oppMinPct=0.5)
