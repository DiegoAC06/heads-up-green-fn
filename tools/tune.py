"""Sweep one poker strategy parameter against the tournament baselines.

Save to tools/tune.py and run from the repository root:
    python -u tools/tune.py --param pf_call_pct --values 0.30,0.35,0.40 --deals 200 --seeds 11
    python -u tools/tune.py --param bet_eq --values 0.55,0.60,0.65 --deals 500 --seeds 11,12 --mode m2

This tool only reports candidates. It NEVER edits strategy.py, and preserves
judge + internal errors so safety-wrapper fallbacks do not hide bugs.
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from evaluator import warmup
from pokerbot.bot import make_bot
from pokerbot.safety import safe_call, safe_choose
from pokerbot.sim import BASELINES, baselineFactories, runMatch
from pokerbot.strategy import DEFAULT_PARAMS


class SafeVariant:
    """Use the same three callback wrappers as agent.py."""

    def __init__(self, params, detection):
        self.bot = make_bot(params=params, detect_opponents=detection)
        self.memory = None

    def new_half(self, config, memory):
        self.memory = memory
        safe_call(self.bot.new_half, config, memory)

    def choose_action(self, obs, memory):
        return safe_choose(self.bot.choose_action, obs, memory)

    def on_hand_end(self, result, memory):
        safe_call(self.bot.on_hand_end, result, memory)


def variant_factory(label, overrides, detection):
    instances = []

    def create():
        instance = SafeVariant(overrides, detection)
        instances.append(instance)
        return instance

    create.botName = label
    create.instances = instances
    return create


def collect(result, candidate, opponent=None):
    """Return error counts, max half-time, and forfeits for a match."""
    stats1 = result.agent_stats["agent1"]
    stats2 = result.agent_stats["agent2"]
    judge_errors = sum(s.errors for s in stats1 + stats2)
    internal = sum(
        int(x.memory.get("internal_errors", 0))
        for factory in (candidate, opponent)
        if factory is not None
        for x in factory.instances
        if x.memory is not None
    )
    max_time = max(s.time_seconds for s in stats1)
    forfeits = sum(bool(getattr(s, "forfeited", False) or getattr(s, "timed_out", False))
                   for s in stats1 + stats2)
    return judge_errors + internal, max_time, forfeits


def main():
    parser = argparse.ArgumentParser(description="Evaluate one parameter at a time")
    parser.add_argument("--param", required=True, choices=sorted(DEFAULT_PARAMS))
    parser.add_argument("--values", required=True, help="Comma-separated numeric trial values")
    parser.add_argument("--seeds", default="11,12", help="Comma-separated paired-deal seeds")
    parser.add_argument("--deals", type=int, default=200)
    parser.add_argument("--mode", choices=("m1", "m2"), default="m2")
    parser.add_argument("--target-time", type=float, default=4.0,
                        help="Preferred maximum seconds per half (default: 4)")
    args = parser.parse_args()
    if args.deals < 1 or args.target_time <= 0:
        parser.error("--deals and --target-time must be positive")
    try:
        values = [float(s) for s in args.values.split(",") if s.strip()]
    except ValueError:
        parser.error("--values must be comma-separated numbers")
    if not values:
        parser.error("At least one trial value is needed")
    seeds = [s.strip() for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        parser.error("At least one seed is needed")

    warmup()
    baselines = baselineFactories()
    detection = args.mode == "m2"
    base_value = DEFAULT_PARAMS[args.param]
    print(f"Mode={args.mode.upper()} parameter={args.param} current={base_value} "
          f"deals={args.deals} seeds={','.join(seeds)}", flush=True)
    print(f"{'value':>9} {'W/C/L':>8} {'sum bb':>10} {'vs default W/C/L':>18} "
          f"{'errors':>8} {'max time':>10} {'eligible':>10}", flush=True)

    eligible = []
    for value in values:
        params = {args.param: value}
        wins = chops = losses = 0
        head_w = head_c = head_l = 0
        score_sum = 0.0
        error_count = forfeits = 0
        maximum_time = 0.0

        for seed in seeds:
            for name in BASELINES:
                candidate = variant_factory(f"trial_{value:g}", params, detection)
                result = runMatch(candidate, baselines[name], seed=seed, deals=args.deals)
                e, seconds, f = collect(result, candidate)
                error_count += e
                forfeits += f
                maximum_time = max(maximum_time, seconds)
                score_sum += result.mean_bb
                if result.verdict == "AGENT1_WIN":
                    wins += 1
                elif result.verdict == "CHOP":
                    chops += 1
                else:
                    losses += 1

            # Direct head-to-head compares this trial against unchanged defaults.
            trial = variant_factory(f"trial_{value:g}", params, detection)
            default = variant_factory("default", None, detection)
            result = runMatch(trial, default, seed=seed, deals=args.deals)
            e, seconds, f = collect(result, trial, default)
            error_count += e
            forfeits += f
            maximum_time = max(maximum_time, seconds)
            if result.verdict == "AGENT1_WIN":
                head_w += 1
            elif result.verdict == "CHOP":
                head_c += 1
            else:
                head_l += 1

        is_eligible = (losses == 0 and head_l == 0 and error_count == 0
                       and forfeits == 0 and maximum_time <= args.target_time)
        print(f"{value:>9.3f} {wins}/{chops}/{losses:<4} {score_sum:>+10.3f} "
              f"{head_w}/{head_c}/{head_l:<14} {error_count:>8} "
              f"{maximum_time:>9.3f}s {'YES' if is_eligible else 'NO':>10}", flush=True)
        if is_eligible:
            eligible.append((wins, score_sum, value))

    if eligible:
        _, score_sum, best = max(eligible)
        print(f"\nBest eligible trial: {args.param}={best:g} (sum bb={score_sum:+.3f}).")
        print("Review more seeds before manually updating pokerbot/strategy.py.")
    else:
        print("\nNo safe candidate met all constraints. Keep the current parameters.")


if __name__ == "__main__":
    main()
