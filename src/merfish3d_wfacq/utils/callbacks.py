from collections.abc import Callable
from typing import Any


def emit_callback(callback: Callable[[str], Any] | None, message: str) -> None:
    """Emit one string message through an optional callback.

    Parameters
    ----------
    callback : Callable[[str], Any] or None
        Callback that accepts one string message.
    message : str
        Message to emit.
    """

    if callback is not None:
        callback(str(message))
