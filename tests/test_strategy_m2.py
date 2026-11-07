
"""Integration and edge-case checks for the M2 opponent detector."""

from types import SimpleNamespace
from unittest.mock import patch

from game import Action, AgentConfig, Game, HandResult, make_deal
from pokerbot.bot import make_bot
from pokerbot.detect import adjust, classify
from pokerbot.opp_model import OppModel
from pokerbot.strategy import DEFAULT_PARAMS


def profile(decisions, raise_rate, fold_to_bet):
    return SimpleNamespace(
        decisions=decisions,
        raise_rate=raise_rate,
        fold_to_bet=fold_to_bet
    )


def test_classifier_labels():
    assert classify(profile(39, 1.0, 0.0)) == "unknown"
    assert classify(profile(40, 0.00, 0.00)) == "station"
    assert classify(profile(40, 0.75, 0.00)) == "maniac"
    assert classify(profile(40, 0.40, 0.25)) == "random"
    assert classify(profile(40, 0.10, 0.70)) == "nit"
    assert classify(profile(40, 0.15, 0.10)) == "solid"


def test_adjust_preserves_base_settings():
    original = DEFAULT_PARAMS.copy()

    assert adjust(original, "unknown") == original
    assert adjust(original, "solid") == original
    assert adjust(original, "station")["bluff_freq"] == 0.0

    # Maniacs now retain M1 settings.
    assert adjust(original, "maniac") == original

    assert adjust(original, "random")["range_step"] == 0.03
    assert adjust(original, "nit")["bluff_freq"] == 0.30

    assert original == DEFAULT_PARAMS
    assert adjust(original, "station") is not original


def test_observer_counts_each_opponent_action_once():
    game = Game(make_deal("observe", 0), button_seat=1)

    # Opponent is on the button and acts first.
    game.apply(Action.CHECK_CALL)
    obs = game.observation(0)

    model = OppModel()
    model.observe(obs)

    assert model.decisions == 1
    assert model.calls == 1

    # Repeated observations must not double-count actions.
    model.observe(obs)

    assert model.decisions == 1
    assert model.calls == 1


def test_unrevealed_opponent_fold_counts_once():
    model = OppModel()

    result = HandResult(
        hand_index=0,
        my_net=2,
        pot=4,
        went_to_showdown=False,
        board=(),
        opp_hole=None,
        winner="me"
    )

    model.hand_end(result)
    model.hand_end(result)

    assert model.folds == model.faced == model.decisions == 1
    assert model.fold_to_bet == 1.0


def test_bot_uses_label_and_resets_each_half():
    bot = make_bot()
    memory = {}

    bot.new_half(AgentConfig(), memory)

    assert memory["label"] == "unknown"
    assert memory["params"] == DEFAULT_PARAMS

    # Simulate a calling-station opponent.
    memory["opp"].decisions = 50
    memory["opp"].faced = 15

    game = Game(
        make_deal("observe", 25),
        button_seat=0,
        hand_index=25
    )
    obs = game.observation(0)

    with patch(
        "pokerbot.bot.decide",
        return_value=Action.CHECK_CALL
    ) as decide_mock:
        result = bot.choose_action(obs, memory)

    assert result == Action.CHECK_CALL
    assert memory["label"] == "station"
    assert decide_mock.call_args.args[2]["bluff_freq"] == 0.0

    # A new half must reset opponent information.
    bot.new_half(AgentConfig(), memory)

    assert memory["opp"].decisions == 0
    assert memory["label"] == "unknown"
    assert memory["params"] == DEFAULT_PARAMS


def test_m1_toggle_retains_default_parameters():
    bot = make_bot(detect_opponents=False)
    memory = {}

    bot.new_half(AgentConfig(), memory)
    memory["opp"].decisions = 50

    game = Game(
        make_deal("observe", 25),
        button_seat=0,
        hand_index=25
    )

    with patch(
        "pokerbot.bot.decide",
        return_value=Action.CHECK_CALL
    ):
        bot.choose_action(game.observation(0), memory)

    assert memory["label"] == "unknown"
    assert memory["params"] == DEFAULT_PARAMS
