
"""Classify opponent tendencies and adjust M1 betting thresholds.

Classifications use only public opponent actions tracked during this half.
The thresholds below are starting values from the M2 strategy plan.
"""


def classify(model):
    """Return a label after enough opponent decisions have been observed."""
    if model.decisions < 40:
        return "unknown"

    if model.raise_rate < 0.05 and model.fold_to_bet < 0.05:
        return "station"

    if model.raise_rate > 0.60 and model.fold_to_bet < 0.10:
        return "maniac"

    if model.raise_rate > 0.25 and model.fold_to_bet > 0.20:
        return "random"

    if model.fold_to_bet > 0.55:
        return "nit"

    return "solid"


def adjust(params, label):
    """Keep M1 settings except against confirmed calling stations."""
    updated = params.copy()

    if label == "station":
        # Calling stations rarely fold, so prioritize value betting.
        # Bluff less and bet strong hands more frequently.
        updated.update(
            bluff_freq=0.0,
            bet_eq=0.55,
            raise_half_eq=0.57,
            range_step=0.0
        )

    if label == "nit":
        # Nits fold most hands to a bet and raise only strong ones:
        # steal more, and call their raises less. (Nightmare Gauntlet, b8/b7.)
        updated.update(
            pf_raise_pct=0.40,
            bluff_freq=0.5,
            bluff_max_eq=0.5,
            pf_call_pct=0.75,
            range_step=0.6,
            call_margin=0.10,
        )

    # Maniac, random, solid, and unknown opponents
    # keep the original M1 settings.
    return updated
