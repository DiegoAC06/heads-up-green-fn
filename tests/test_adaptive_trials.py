"""Adaptive sampling: few samples for clear decisions, more near a threshold."""

import time

from game import Action, Game, make_deal
from pokerbot.strategy import DEFAULT_PARAMS, decide, nearest_threshold_gap
from pokerbot.timebank import TimeBank


def _flopFacingBet(deal=0):
  game = Game(make_deal("adapt", deal), button_seat=0, hand_index=deal)
  game.apply(Action.CHECK_CALL)
  game.apply(Action.CHECK_CALL)
  game.apply(Action.CHECK_CALL)
  game.apply(Action.RAISE_HALF)
  return game.observation(1)


def _memory(spent=0.0):
  tb = TimeBank(8.0, 2.0, 0.015)
  tb.usedS = spent
  tb.begin()
  return {"tb": tb, "starting_stack": 80}


def _calls(monkeypatch, eq, params, memory=None):
  import pokerbot.strategy as strat
  calls = []
  monkeypatch.setattr(strat, "equity", lambda h, b, t, d, mp, s, **k: calls.append(t) or eq)
  decide(_flopFacingBet(), memory or _memory(), params)
  return calls


def testGapUsesFacingBetThreshold():
  obs = _flopFacingBet()
  potOdds = obs.to_call / (obs.pot + obs.to_call)
  assert abs(nearest_threshold_gap(potOdds + DEFAULT_PARAMS["call_margin"], obs, DEFAULT_PARAMS)) < 1e-9


def testClearDecisionUsesPilotOnly(monkeypatch):
  assert _calls(monkeypatch, 0.95, dict(DEFAULT_PARAMS, adaptive_trials=True)) == [120]


def testCloseDecisionAddsSamples(monkeypatch):
  obs = _flopFacingBet()
  eq = obs.to_call / (obs.pot + obs.to_call) + DEFAULT_PARAMS["call_margin"] + 0.01
  assert _calls(monkeypatch, eq, dict(DEFAULT_PARAMS, adaptive_trials=True)) == [120, 600]


def testStarvedDeadlineNoExtra(monkeypatch):
  obs = _flopFacingBet()
  eq = obs.to_call / (obs.pot + obs.to_call) + DEFAULT_PARAMS["call_margin"]
  assert len(_calls(monkeypatch, eq, dict(DEFAULT_PARAMS, adaptive_trials=True), _memory(spent=7.5))) == 1


def testSwitchOffOneCallOf300(monkeypatch):
  assert _calls(monkeypatch, 0.5, dict(DEFAULT_PARAMS, adaptive_trials=False)) == [300]
