"""Classify opponent tendencies and adapt M1 betting thresholds.

Uses only the public actions tracked in the current half. Starting thresholds
come from the M2 plan; keep M1 settings against 'maniac' while testing whether
aggressive-opponent adaptations actually improve results.
"""


def classify(model):
    """Classify only after at least 40 observed opponent decisions."""
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
    """Return adapted parameters without mutating the base M1 settings."""
    updated = params.copy()

    if label == "station":
        # Bet good hands for value; avoid bluffs against frequent callers.
        updated.update(
            bluff_freq=0.0,
            bet_eq=0.55,
            raise_half_eq=0.57,
            range_step=0.0,
        )
    elif label == "maniac":
        # Preserve the winning M1 settings against frequent raisers.
        # The previous M2 overrides underperformed M1 on seeds 11 and 12.
        pass
    elif label == "random":
        # Use a wider estimated range and bluff less frequently.
        updated.update(
            range_step=0.03,
            bluff_freq=0.03,
            pf_call_pct=0.30,
        )
    elif label == "nit":
        # Raise and bluff more against opponents that fold frequently.
        updated.update(
            bluff_freq=0.30,
            bluff_max_eq=0.45,
            pf_raise_pct=0.55,
        )

    # 'solid' and 'unknown' also keep the unmodified M1 parameters.
    return updated
