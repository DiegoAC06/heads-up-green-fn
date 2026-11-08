"""Fold-equity bluffing: bluff in proportion to how often they fold to a bet on this street."""

from game import Action, Game, HandResult, Street, make_deal
from pokerbot.opp_model import OppModel
from pokerbot.strategy import DEFAULT_PARAMS, decide
from pokerbot.timebank import TimeBank


def _flopCheckedToUsOnButton(deal):
  # we are the button (seat 0): limp, they check, they check the flop to us
  game = Game(make_deal("fe", deal), button_seat=0, hand_index=deal)
  game.apply(Action.CHECK_CALL)
  game.apply(Action.CHECK_CALL)
  game.apply(Action.CHECK_CALL)
  return game.observation(0)


def _memory(foldRate):
  m = OppModel()
  m.faced, m.folds = 100, int(100 * foldRate)
  m.streetFaced = {Street.FLOP: 100, Street.TURN: 0, Street.RIVER: 0}
  m.streetFolds = {Street.FLOP: int(100 * foldRate), Street.TURN: 0, Street.RIVER: 0}
  tb = TimeBank(8.0, 2.0, 0.015)
  tb.begin()
  return {"opp": m, "tb": tb, "starting_stack": 80}


def _bluffs(monkeypatch, params, foldRate, deals=200):
  import pokerbot.strategy as strat
  monkeypatch.setattr(strat, "equity", lambda *a, **k: 0.2)   # always a weak hand
  return sum(decide(_flopCheckedToUsOnButton(d), _memory(foldRate), params) == Action.RAISE_HALF for d in range(deals))


def testFoldToBetOnPrior():
  assert abs(OppModel().foldToBetOn(Street.TURN) - 0.35) < 1e-9


def testFoldToBetOnCountsPerStreet():
  m = OppModel()
  m.hand_end(HandResult(hand_index=1, my_net=6, pot=12, went_to_showdown=False, board=(1, 2, 3, 4),
                        opp_hole=None, winner="me"))
  assert m.streetFaced[Street.TURN] == 1 and m.streetFolds[Street.TURN] == 1


def testNoBluffsVsStation(monkeypatch):
  assert _bluffs(monkeypatch, dict(DEFAULT_PARAMS, fe_bluff=True, bluff_freq=0.0), 0.02) == 0


def testBluffsVsFolder(monkeypatch):
  assert _bluffs(monkeypatch, dict(DEFAULT_PARAMS, fe_bluff=True), 0.8) > 40


def testSwitchOffMatchesBaseline(monkeypatch):
  off = dict(DEFAULT_PARAMS, fe_bluff=False)
  assert _bluffs(monkeypatch, off, 0.8) < 40   # only the old 8% button bluff
  assert _bluffs(monkeypatch, off, 0.8) == _bluffs(monkeypatch, off, 0.02)   # fold rate unused when off
