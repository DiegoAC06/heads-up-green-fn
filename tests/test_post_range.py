"""Opponent postflop aggression -> range of hands they bet with."""

from game import Action, Game, HandResult, Street, make_deal
from pokerbot.opp_model import OppModel
from pokerbot.strategy import DEFAULT_PARAMS, decide
from pokerbot.timebank import TimeBank


def _flopFacingBet(deal):
  # button (seat 0) limps, we check; on the flop we check and the opponent bets
  game = Game(make_deal("post", deal), button_seat=0)
  game.apply(Action.CHECK_CALL)
  game.apply(Action.CHECK_CALL)
  game.apply(Action.CHECK_CALL)     # seat 1 (us) checks the flop
  game.apply(Action.RAISE_HALF)     # opponent bets
  return game.observation(1)


def _memory(postRaises, postDecisions):
  m = OppModel()
  m.postRaises, m.postDecisions = postRaises, postDecisions
  tb = TimeBank(8.0, 2.0, 0.015)
  tb.begin()
  return {"opp": m, "tb": tb, "starting_stack": 80}


def _capturedFrac(monkeypatch, params, memory, obs):
  import pokerbot.strategy as strat
  seen = []
  monkeypatch.setattr(strat, "equity", lambda h, b, t, d, mp, s, oppTopFrac=1.0: seen.append(oppTopFrac) or 0.5)
  decide(obs, memory, params)
  return seen[-1]


def testPafStartsAtPriorAndCountsPostflopOnly():
  m = OppModel()
  assert abs(m.paf - 0.30) < 1e-9
  game = Game(make_deal("post", 1), button_seat=0)
  game.apply(Action.RAISE_HALF)      # opponent preflop raise: not postflop
  game.apply(Action.CHECK_CALL)
  game.apply(Action.CHECK_CALL)      # we check flop
  game.apply(Action.RAISE_HALF)      # opponent flop bet
  m.observe(game.observation(1))
  assert (m.postRaises, m.postDecisions) == (1, 1)


def testPostflopFoldEndingTheHandCounts():
  m = OppModel()
  m.hand_end(HandResult(hand_index=4, my_net=6, pot=12, went_to_showdown=False,
                        board=(1, 2, 3), opp_hole=None, winner="me"))
  assert m.postDecisions == 1 and m.postRaises == 0


def testBetFromTightBettorNarrowsRange(monkeypatch):
  obs = _flopFacingBet(0)
  assert obs.street == Street.FLOP and obs.to_call > 0
  got = _capturedFrac(monkeypatch, DEFAULT_PARAMS, _memory(20, 100), obs)
  assert got < 0.5


def testManiacBetsDoNotNarrow(monkeypatch):
  got = _capturedFrac(monkeypatch, DEFAULT_PARAMS, _memory(95, 100), _flopFacingBet(0))
  assert got >= 0.8   # (95 + 0.3 x 20) / 120 = 0.84: barely narrowed


def testSwitchOff(monkeypatch):
  params = dict(DEFAULT_PARAMS, post_range=False)
  assert _capturedFrac(monkeypatch, params, _memory(20, 100), _flopFacingBet(0)) == 1.0


def testRiverOnlyBettorIsNotNarrowedOnTheRiver(monkeypatch):
  # bets almost every river but nothing else: a river bet tells us little
  m = OppModel()
  m.postRaises, m.postDecisions = 100, 300
  m.streetRaises = {Street.FLOP: 0, Street.TURN: 0, Street.RIVER: 100}
  m.streetDecisions = {Street.FLOP: 100, Street.TURN: 100, Street.RIVER: 100}
  assert m.pafOn(Street.RIVER) > 0.85
  assert m.pafOn(Street.FLOP) < 0.15


def testPafOnShrinksTowardOverallWithFewHands():
  m = OppModel()
  m.postRaises, m.postDecisions = 50, 100
  m.streetRaises = {Street.FLOP: 1, Street.TURN: 0, Street.RIVER: 0}
  m.streetDecisions = {Street.FLOP: 1, Street.TURN: 0, Street.RIVER: 0}
  # one flop bet in one flop decision should not jump to 1.0
  assert 0.45 < m.pafOn(Street.FLOP) < 0.6
