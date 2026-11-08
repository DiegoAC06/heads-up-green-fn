"""Preflop opens and steals on the button."""

from game import Action, Game, HandResult, make_deal
from pokerbot.bot import make_bot
from pokerbot.strategy import DEFAULT_PARAMS, decide, pf_fold_to_raise
from pokerbot.timebank import TimeBank


def _btnUnopened(deal):
  game = Game(make_deal("open", deal), button_seat=1, hand_index=deal)
  return game.observation(1)


def _memory(raises=0, folds=0):
  tb = TimeBank(8.0, 2.0, 0.015)
  tb.begin()
  return {"tb": tb, "starting_stack": 80, "ourPfRaises": raises, "ourPfRaiseFolds": folds}


def testPfFoldToRaisePrior():
  assert abs(pf_fold_to_raise({}) - 0.35) < 1e-9


def testCountsFoldToOurRaise():
  bot = make_bot()
  memory = {}
  from game import AgentConfig
  bot.new_half(AgentConfig(), memory)
  memory["ourPfRaises"] = 1
  memory["pfLast"] = (7, True)
  bot.on_hand_end(HandResult(hand_index=7, my_net=2, pot=6, went_to_showdown=False, board=(),
                             opp_hole=None, winner="me"), memory)
  assert memory["ourPfRaiseFolds"] == 1


def testStealsOnlyVsFolders():
  params = dict(DEFAULT_PARAMS, steal_fold_min=0.5, steal_pct=0.30)
  raisesVsFolder = sum(decide(_btnUnopened(d), _memory(100, 80), params) in (Action.RAISE_POT, Action.RAISE_HALF) for d in range(200))
  raisesVsCaller = sum(decide(_btnUnopened(d), _memory(100, 10), params) in (Action.RAISE_POT, Action.RAISE_HALF) for d in range(200))
  assert raisesVsFolder > raisesVsCaller + 50


def testSwitchOffMatchesBaseline():
  off = dict(DEFAULT_PARAMS, btn_open_pct=0.80, steal_fold_min=1.1)
  for d in range(200):
    obs = _btnUnopened(d)
    assert decide(obs, _memory(100, 80), off) == decide(obs, _memory(), off)
