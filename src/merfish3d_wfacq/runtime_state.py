from typing import Any
from uuid import uuid4

_RUNTIME_STATE: dict[str, dict[str, Any]] = {}


def register_runtime_state(state: dict[str, Any]) -> str:
    """Register one in-memory runtime state payload.

    Parameters
    ----------
    state : dict[str, Any]
        Mutable runtime state shared across prepared MERFISH components.

    Returns
    -------
    str
        Stable identifier used to retrieve the registered state.
    """

    state_id = str(uuid4())
    _RUNTIME_STATE[state_id] = state
    return state_id


def get_runtime_state(state_id: str) -> dict[str, Any]:
    """Return previously registered runtime state.

    Parameters
    ----------
    state_id : str
        Identifier returned by :func:`register_runtime_state`.

    Returns
    -------
    dict[str, Any]
        Registered runtime state payload.
    """

    return _RUNTIME_STATE[state_id]


def create_drift_reference_runtime(
    reference_tile: int,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Create shared in-memory drift runtime state and setup payload.

    Parameters
    ----------
    reference_tile : int
        Tile index used as the round-1 drift reference.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str]]
        Shared runtime state plus the JSON-serializable setup payload stamped
        onto the MERFISH setup action.
    """

    store = {"reference_tile": int(reference_tile), "frames": []}
    return store, {"drift_reference_store_id": register_runtime_state(store)}


def unregister_runtime_state(state_id: str | None) -> None:
    """Remove registered runtime state if present.

    Parameters
    ----------
    state_id : str or None
        Identifier returned by :func:`register_runtime_state`.
    """

    if state_id is not None:
        _RUNTIME_STATE.pop(state_id, None)
