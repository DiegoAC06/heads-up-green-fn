"""In-process duplicate match simulator.

InProcessHandle stands in for the judge's HttpAgentHandle, so the judge's own
play_half, combine_halves and format_report do the dealing and scoring. No
servers are needed.

Usage:
  python -m pokerbot.sim agent.py baseline-bots/baseline_random.py --deals 1000 --seed 7
  python -m pokerbot.sim agent.py --gauntlet --deals 500
"""

import argparse
import importlib.util
import itertools
import os
import sys
import time
import traceback
from types import SimpleNamespace
from typing import Any, Callable, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from game import ERROR_LIMIT, Action, AgentConfig, HandResult, Observation, default_action  # noqa: E402
from judge_engine import (  # noqa: E402
  MAX_STORED_ERRORS, AgentStats, MatchResult, coerce_action, combine_halves, format_report, play_half,
)

BASELINES = ("baseline_random", "baseline_always_half", "baseline_always_pot", "baseline_calling_station")
_moduleCounter = itertools.count()


def botFromFile(path: str) -> Callable[[], Any]:
  """Factory that loads a fresh copy of a bot file (new module each half)."""
  fullPath = os.path.abspath(path)
  if not os.path.isfile(fullPath):
    raise FileNotFoundError(fullPath)

  def factory() -> Any:
    name = f"simbot_{next(_moduleCounter)}"
    spec = importlib.util.spec_from_file_location(name, fullPath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

  factory.botName = os.path.splitext(os.path.basename(fullPath))[0]
  return factory


def botFromFunctions(chooseAction: Callable, newHalf: Optional[Callable] = None,
                     onHandEnd: Optional[Callable] = None) -> Callable[[], Any]:
  """Factory for a bot made of plain functions, e.g. a strategy variant."""
  def factory() -> Any:
    return SimpleNamespace(
      new_half=newHalf or (lambda config, memory: None),
      choose_action=chooseAction,
      on_hand_end=onHandEnd or (lambda result, memory: None),
    )

  factory.botName = getattr(chooseAction, "__name__", "functions")
  return factory


def _toAnswer(action: Any) -> Any:
  # same conversion agent.py's plumbing does before the judge reads it
  if isinstance(action, Action):
    return action.name
  if action is None or isinstance(action, (str, int, float, bool)):
    return action
  return repr(action)


class InProcessHandle:
  """Duck-types judge_engine.HttpAgentHandle for play_half."""

  def __init__(self, factory: Callable[[], Any], name: str, config: Optional[AgentConfig] = None,
               faithful: bool = True) -> None:
    self.factory = factory
    self.name = name
    self.config = config or AgentConfig()
    self.faithful = faithful
    self.stats = AgentStats()
    self.bot: Any = None
    self.memory: dict = {}

  def _recordError(self, message: str) -> None:
    self.stats.errors += 1
    if len(self.stats.error_messages) < MAX_STORED_ERRORS:
      self.stats.error_messages.append(message)
    if self.stats.errors >= ERROR_LIMIT:
      self.stats.forfeited = True

  def _usable(self) -> bool:
    if self.stats.forfeited:
      return False
    if self.config.time_bank_seconds - self.stats.time_seconds <= 0:
      self.stats.timed_out = True
      self.stats.skipped_calls += 1
      return False
    return True

  def _timed(self, label: str, fn: Callable, *args: Any) -> tuple[bool, Any]:
    """Run fn, charge its time, and turn an exception into an error."""
    self.stats.calls += 1
    start = time.perf_counter()
    try:
      return True, fn(*args)
    except Exception as exc:
      self._recordError(f"{label} raised {type(exc).__name__}: {exc}\n{traceback.format_exc()}")
      return False, None
    finally:
      elapsed = time.perf_counter() - start
      self.stats.time_seconds += elapsed
      self.stats.wall_seconds += elapsed

  # the methods below are called by judge_engine.play_half, so they keep its names

  def start_half(self) -> None:
    self.stats = AgentStats()
    self.memory = {}
    try:
      self.bot = self.factory()
    except Exception as exc:
      self._recordError(f"bot failed to load: {exc}\n{traceback.format_exc()}")
      self.stats.forfeited = True
      return
    ok, _ = self._timed("new_half", self.bot.new_half, AgentConfig.from_dict(self.config.to_dict()), self.memory)
    if not ok:
      self.stats.forfeited = True

  def act(self, obs: Observation) -> Action:
    fallback = default_action(obs.to_call)
    if not self._usable():
      return fallback
    seen = Observation.from_dict(obs.to_dict()) if self.faithful else obs
    ok, raw = self._timed("choose_action", self.bot.choose_action, seen, self.memory)
    if not ok:
      return fallback
    action = coerce_action(_toAnswer(raw))
    if action is None:
      self._recordError(f"choose_action returned {raw!r}, which is not an Action")
      return fallback
    if action not in obs.legal_actions:
      legal = ", ".join(a.name for a in obs.legal_actions)
      self._recordError(f"choose_action returned illegal action {action.name} (legal: {legal})")
      return fallback
    return action

  def hand_end(self, result: HandResult) -> None:
    if not self._usable():
      return
    seen = HandResult.from_dict(result.to_dict()) if self.faithful else result
    self._timed("on_hand_end", self.bot.on_hand_end, seen, self.memory)

  def flush_hand_results(self) -> None:
    # results are delivered immediately in hand_end, so nothing is queued
    pass

  def describe(self) -> str:
    return self.name


def _name(factory: Callable, fallback: str) -> str:
  return getattr(factory, "botName", fallback)


def runMatch(factoryA: Callable, factoryB: Callable, seed: str = "1", deals: int = 1000,
             keepLogs: bool = False) -> MatchResult:
  """Both halves of a duplicate match, scored exactly like the judge."""
  a = InProcessHandle(factoryA, _name(factoryA, "agent1"))
  b = InProcessHandle(factoryB, _name(factoryB, "agent2"))
  half1 = play_half(a, b, seed, 1, num_deals=deals, keep_logs=keepLogs)
  half2 = play_half(a, b, seed, 2, num_deals=deals, keep_logs=keepLogs)
  return combine_halves(half1, half2)


def baselineFactories() -> dict[str, Callable]:
  return {name: botFromFile(os.path.join(ROOT, "baseline-bots", f"{name}.py")) for name in BASELINES}


def runGauntlet(factory: Callable, seeds: tuple = ("1", "2"), deals: int = 1000) -> dict[str, list[MatchResult]]:
  """Play the factory against every baseline for every seed."""
  results: dict[str, list[MatchResult]] = {}
  for name, opponent in baselineFactories().items():
    results[name] = [runMatch(factory, opponent, seed=str(s), deals=deals) for s in seeds]
  return results


def maxTimeS(result: MatchResult, agent: str = "agent1") -> float:
  return max(s.time_seconds for s in result.agent_stats[agent])


def gauntletLines(results: dict[str, list[MatchResult]], seeds: tuple) -> list[str]:
  lines = [f"{'opponent':<26}{'seed':>6}{'mean_bb':>10}  {'ci':<20}{'verdict':<14}{'errors':>7}{'max_time_s':>12}"]
  for name, matches in results.items():
    for seed, r in zip(seeds, matches):
      errors = sum(s.errors for s in r.agent_stats["agent1"])
      ci = f"[{r.ci[0]:+.3f}, {r.ci[1]:+.3f}]"
      lines.append(f"{name:<26}{seed:>6}{r.mean_bb:>+10.3f}  {ci:<20}{r.verdict:<14}{errors:>7}{maxTimeS(r):>12.3f}")
  return lines


def main(argv: Optional[list] = None) -> int:
  p = argparse.ArgumentParser(description="in-process duplicate match simulator")
  p.add_argument("botA", help="path to agent1's bot file")
  p.add_argument("botB", nargs="?", help="path to agent2's bot file")
  p.add_argument("--gauntlet", action="store_true", help="play botA against all four baselines")
  p.add_argument("--deals", type=int, default=1000)
  p.add_argument("--seed", default="1", help="seed, or comma separated seeds for --gauntlet")
  args = p.parse_args(argv)

  a = botFromFile(args.botA)
  if args.gauntlet:
    seeds = tuple(args.seed.split(","))
    print("\n".join(gauntletLines(runGauntlet(a, seeds=seeds, deals=args.deals), seeds)))
    return 0
  if not args.botB:
    p.error("give a second bot file, or use --gauntlet")
  print(format_report(runMatch(a, botFromFile(args.botB), seed=args.seed, deals=args.deals)))
  return 0


if __name__ == "__main__":
  sys.exit(main())
