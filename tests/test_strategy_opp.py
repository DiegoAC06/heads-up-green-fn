"""Opponent-model tests for the TAMU heads-up poker bot.

Save as tests/test_strategy_opp.py. The baseline checks use the real in-process
judge. No HTTP servers are needed.
"""

from game import Action, Game, HandResult, make_deal
from pokerbot.detect import classify
from pokerbot.opp_model import OppModel


def test_repeated_observation_does_not_double_count():
    game = Game(make_deal("opp-test", 0), button_seat=1)
    game.apply(Action.RAISE_POT)
    obs = game.observation(0)
    model = OppModel()

    model.observe(obs)
    assert model.decisions == 1
    assert model.raises == 1
    assert model.raises_this_hand(obs) == 1

    model.observe(obs)
    assert model.decisions == 1
    assert model.raises == 1


def test_folded_opponent_without_revealed_cards():
    result = HandResult(
        hand_index=3, my_net=2, pot=4,
        went_to_showdown=False, board=(), opp_hole=None, winner="me",
    )
    model = OppModel()
    model.hand_end(result)
    model.hand_end(result)
    assert model.decisions == 1
    assert model.folds == 1
    assert model.faced == 1
    assert model.fold_to_bet == 1.0


def test_model_handles_free_checks_without_counting_calls():
    game = Game(make_deal("opp-test", 0), button_seat=1)
    game.apply(Action.CHECK_CALL)  # Opponent calls the blind, not a free check.
    model = OppModel()
    model.observe(game.observation(0))
    assert model.decisions == 1
    assert model.calls == 1
    assert model.raises == 0


def _profile_against(opponent_name, deals=200):
    """Return both per-half opponent models from a faithful duplicate match."""
    import pytest
    pytest.importorskip("judge_engine")
    from evaluator import warmup
    from pokerbot.bot import make_bot
    from pokerbot.safety import safe_call, safe_choose
    from pokerbot.sim import baselineFactories, runMatch

    warmup()
    models = []
    memories = []

    class TrackingBot:
        def __init__(self):
            self.bot = make_bot(detect_opponents=True)

        def new_half(self, config, memory):
            safe_call(self.bot.new_half, config, memory)
            models.append(memory["opp"])
            memories.append(memory)

        def choose_action(self, obs, memory):
            return safe_choose(self.bot.choose_action, obs, memory)

        def on_hand_end(self, result, memory):
            safe_call(self.bot.on_hand_end, result, memory)

    def factory():
        return TrackingBot()

    factory.botName = "tracking_bot"
    match = runMatch(factory, baselineFactories()[opponent_name], seed="model-test", deals=deals)
    assert sum(s.errors for side in match.agent_stats.values() for s in side) == 0
    assert all(m["internal_errors"] == 0 for m in memories)
    assert len(models) == 2
    return models


def test_always_pot_profile():
    for model in _profile_against("baseline_always_pot"):
        assert model.decisions >= 40
        assert model.raise_rate > 0.80
        assert model.fold_to_bet < 0.05
        assert classify(model) == "maniac"


def test_calling_station_profile():
    for model in _profile_against("baseline_calling_station"):
        assert model.decisions >= 40
        assert model.raise_rate < 0.05
        assert model.fold_to_bet < 0.05
        assert classify(model) == "station"


def test_random_opponent_has_mixed_actions():
    for model in _profile_against("baseline_random"):
        assert model.decisions >= 40
        assert model.raises > 0
        assert model.calls > 0
