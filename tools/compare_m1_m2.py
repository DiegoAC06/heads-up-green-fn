"""Compare M1 and M2 with the same safety wrappers used by agent.py.

Save as tools/compare_m1_m2.py. Run from the repository root:
    python -u tools/compare_m1_m2.py --deals 100 --seeds 11
    python -u tools/compare_m1_m2.py --deals 500 --seeds 11,12
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


class WrappedVariant:
    """Match the three tournament callbacks in agent.py."""

    def __init__(self, detection_enabled):
        self.bot = make_bot(detect_opponents=detection_enabled)
        self.memory = None

    def new_half(self, config, memory):
        self.memory = memory
        safe_call(self.bot.new_half, config, memory)

    def choose_action(self, obs, memory):
        return safe_choose(self.bot.choose_action, obs, memory)

    def on_hand_end(self, result, memory):
        safe_call(self.bot.on_hand_end, result, memory)


def variant_factory(name, detection_enabled):
    """Create a fresh bot for each half, retaining instances for diagnostics."""
    instances = []

    def factory():
        instance = WrappedVariant(detection_enabled)
        instances.append(instance)
        return instance

    factory.botName = name
    factory.instances = instances
    return factory


def count_internal_errors(factory):
    return sum(
        instance.memory.get("internal_errors", 0)
        for instance in factory.instances
        if instance.memory is not None
    )


def report(label, result, agent1_factory, agent2_factory=None):
    stats1 = result.agent_stats["agent1"]
    stats2 = result.agent_stats["agent2"]
    judge_errors1 = sum(s.errors for s in stats1)
    judge_errors2 = sum(s.errors for s in stats2)
    internal_errors1 = count_internal_errors(agent1_factory)
    internal_errors2 = count_internal_errors(agent2_factory) if agent2_factory else 0
    max_time = max(s.time_seconds for s in stats1)
    ci = result.ci

    print(
        f"{label:<34} {result.mean_bb:+8.3f} "
        f"[{ci[0]:+.3f}, {ci[1]:+.3f}] {result.verdict:<13} "
        f"judge_errors={judge_errors1}/{judge_errors2} "
        f"internal_errors={internal_errors1}/{internal_errors2} "
        f"max_time={max_time:.3f}s",
        flush=True,
    )

    if judge_errors1 or judge_errors2 or internal_errors1 or internal_errors2:
        # The old script hid the reason for the forfeits. Show it now.
        for side, stats in (("agent1", stats1), ("agent2", stats2)):
            for half_index, stat in enumerate(stats, start=1):
                if stat.error_messages:
                    print(f"  {side}, half {half_index}: {stat.error_messages[0]}", flush=True)

        for side, factory in (("agent1", agent1_factory), ("agent2", agent2_factory)):
            if factory is None:
                continue
            for half_index, instance in enumerate(factory.instances, start=1):
                if instance.memory and instance.memory.get("internal_errors", 0):
                    print(
                        f"  {side}, half {half_index} internal: "
                        f"{instance.memory.get('last_internal_error')}",
                        flush=True,
                    )
        raise RuntimeError("Bot errors detected; fix them before judging M1/M2 performance")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deals", type=int, default=100)
    parser.add_argument("--seeds", default="11")
    args = parser.parse_args()
    if args.deals < 1:
        parser.error("--deals must be positive")

    seeds = [seed.strip() for seed in args.seeds.split(",") if seed.strip()]
    if not seeds:
        parser.error("provide at least one seed")

    # agent.py does this during startup; our in-process factory must too.
    warmup()
    baselines = baselineFactories()

    for seed in seeds:
        print(f"\n=== Seed {seed}; {args.deals} paired deals ===", flush=True)
        for opponent_name in BASELINES:
            for name, enabled in (("M1", False), ("M2", True)):
                agent = variant_factory(name, enabled)
                result = runMatch(agent, baselines[opponent_name], seed=seed, deals=args.deals)
                report(f"{name} vs {opponent_name}", result, agent)

        m2 = variant_factory("M2", True)
        m1 = variant_factory("M1", False)
        result = runMatch(m2, m1, seed=seed, deals=args.deals)
        report("M2 vs M1 (head-to-head)", result, m2, m1)


if __name__ == "__main__":
    main()
