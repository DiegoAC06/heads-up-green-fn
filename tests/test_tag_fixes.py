"""Tests for the tight-aggressive defenses (preflop raise range from observed raise rate)."""

from game import Action, Game, Street, make_deal
from pokerbot.opp_model import OppModel
from pokerbot.strategy import DEFAULT_PARAMS, decide
from pokerbot.timebank import TimeBank


def _memoryWithPfr(raises, opps):
  model = OppModel()
  model.pfRaises, model.pfOpps = raises, opps
  tb = TimeBank(8.0, 2.0, 0.015)
  tb.begin()
  return {"opp": model, "tb": tb, "starting_stack": 80}


def testPfrStartsAtPriorAndShrinks():
  m = OppModel()
  assert abs(m.pfr - 0.25) < 1e-9
  m.pfRaises, m.pfOpps = 80, 100
  assert 0.6 < m.pfr < 0.8


def testObserveCountsOnePreflopOpportunityPerHand():
  # opponent (button) raises, we re-raise, they raise again: one opportunity, one raise
  game = Game(make_deal("pfr", 0), button_seat=0)
  game.apply(Action.RAISE_HALF)
  game.apply(Action.RAISE_HALF)
  game.apply(Action.RAISE_HALF)
  m = OppModel()
  m.observe(game.observation(1))
  assert (m.pfOpps, m.pfRaises) == (1, 1)
  m.observe(game.observation(1))  # same history again: no double count
  assert (m.pfOpps, m.pfRaises) == (1, 1)


def testOpponentFoldBeforeActingPreflopCountsAsOpportunity():
  from game import HandResult
  m = OppModel()
  m.hand_end(HandResult(hand_index=3, my_net=2, pot=6, went_to_showdown=False, board=(), opp_hole=None, winner="me"))
  assert (m.pfOpps, m.pfRaises) == (1, 0)


def _flopAfterPreflopRaise(deal):
  # opponent (button, seat 0) raises preflop, we call; we act first on the flop
  game = Game(make_deal("tag", deal), button_seat=0)
  game.apply(Action.RAISE_POT)
  game.apply(Action.CHECK_CALL)
  return game.observation(1)


def _capturedRange(monkeypatch, params, memory, obs):
  import pokerbot.strategy as strat
  seen = []
  monkeypatch.setattr(strat, "equity", lambda h, b, t, d, oppMinPct, s, **kw: seen.append(oppMinPct) or 0.5)
  decide(obs, memory, params)
  return seen[-1]


def testPostflopUsesInferredRaisingRange(monkeypatch):
  obs = _flopAfterPreflopRaise(0)
  assert obs.street == Street.FLOP
  # 28 raises in 100 hands smooths (prior 0.25 x 20) to 0.275 -> assumed range pct >= 0.725
  got = _capturedRange(monkeypatch, DEFAULT_PARAMS, _memoryWithPfr(28, 100), obs)
  assert abs(got - 0.725) < 1e-9


def testManiacRaiseDoesNotNarrowRange(monkeypatch):
  obs = _flopAfterPreflopRaise(0)
  got = _capturedRange(monkeypatch, DEFAULT_PARAMS, _memoryWithPfr(100, 100), obs)
  assert got < 0.2   # only the old per-raise step applies


def testSwitchOffKeepsOldRange(monkeypatch):
  obs = _flopAfterPreflopRaise(0)
  params = dict(DEFAULT_PARAMS, pf_range_post=False)
  got = _capturedRange(monkeypatch, params, _memoryWithPfr(28, 100), obs)
  assert abs(got - DEFAULT_PARAMS["range_step"]) < 1e-9
