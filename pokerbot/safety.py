from game import Action

def record_error(memory, message):
    # Track failures within this half, even when a fallback prevents
    # the judge from seeing an exception or illegal action.
    memory["internal_errors"] = memory.get("internal_errors", 0) + 1

    # Keep the latest error message to help identify what went wrong.
    memory["last_internal_error"] = message


def safe_choose(fn, obs, memory):
    # Run the strategy function and check its result before returning
    # an action to the judge.
    try:
        action = fn(obs, memory)

        # Require an actual Action enum and confirm that the action
        # is allowed in the current game state.
        if isinstance(action, Action) and action in obs.legal_actions:
            return action

        # The strategy returned an invalid value or an unavailable action.
        record_error(memory, f"Illegal action: {action!r}")

    except Exception as exc:
        # Save the exception instead of letting it reach the judge.
        record_error(memory, f"{type(exc).__name__}: {exc}")

    # CHECK_CALL is always legal in this game. It checks when free
    # and calls otherwise, so this fallback can still cost chips.
    # This wrapper does not prevent timeouts.
    return Action.CHECK_CALL


def safe_call(fn, data, memory):
    # Run a function that does not return an action:
    # new_half receives config; on_hand_end receives a hand result.
    try:
        fn(data, memory)

    except Exception as exc:
        # Record the failure without passing the exception to the judge.
        # Changes made before the exception are not automatically undone.
        record_error(memory, f"{type(exc).__name__}: {exc}")