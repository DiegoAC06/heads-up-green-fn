import os

from game import Action
from pokerbot.sim import botFromFile, botFromFunctions, runMatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATION = botFromFile(os.path.join(ROOT, "baseline-bots", "baseline_calling_station.py"))
POT = botFromFile(os.path.join(ROOT, "baseline-bots", "baseline_always_pot.py"))
CHECK_FOLD = botFromFunctions(lambda obs, m: Action.CHECK_CALL if obs.to_call == 0 else Action.FOLD)


def testIdenticalBotsTieEveryDeal():
  r = runMatch(STATION, STATION, seed="t", deals=100)
  assert all(x == 0 for x in r.paired_bb)
  assert r.verdict == "CHOP"


def testAlwaysPotBeatsCheckFold():
  assert runMatch(CHECK_FOLD, POT, seed="t", deals=300).verdict == "AGENT2_WIN"


def testCrashingBotForfeitsAndLoses():
  def boom(obs, m):
    raise ValueError("x")
  r = runMatch(botFromFunctions(boom), STATION, seed="t", deals=50)
  assert r.agent_stats["agent1"][0].forfeited
  assert r.verdict == "AGENT2_WIN"


def testIllegalActionIsAnError():
  # fold is illegal whenever checking is free
  r = runMatch(botFromFunctions(lambda o, m: Action.FOLD), STATION, seed="t", deals=50)
  assert r.agent_stats["agent1"][0].errors >= 1


def testMemoryIsFreshEachHalf():
  def newHalf(cfg, m):
    assert m == {}
    m["n"] = 0

  def choose(obs, m):
    m["n"] += 1
    return Action.CHECK_CALL

  r = runMatch(botFromFunctions(choose, newHalf=newHalf), STATION, seed="t", deals=20)
  assert not r.agent_stats["agent1"][0].forfeited
  assert not r.agent_stats["agent1"][1].forfeited


def testOnHandEndCrashCountsAsError():
  def badEnd(result, m):
    raise KeyError("y")
  r = runMatch(botFromFunctions(lambda o, m: Action.CHECK_CALL, onHandEnd=badEnd), STATION, seed="t", deals=20)
  assert r.agent_stats["agent1"][0].forfeited
