from contextlib import contextmanager
from .types import TropicalMode


_global_tropical_mode: TropicalMode = TropicalMode.MIN_PLUS


def set_global_tropical_mode(mode: TropicalMode):
    """
    Set the global tropical mode for the algebra operations.

    :param mode: The TropicalMode to set as global.
    """
    if mode not in [TropicalMode.MIN_PLUS, TropicalMode.MAX_PLUS]:
        raise ValueError(
            "Invalid tropical mode. Use TropicalMode.MIN_PLUS or TropicalMode.MAX_PLUS."
        )
    global _global_tropical_mode
    _global_tropical_mode = mode


def get_global_tropical_mode() -> TropicalMode:
    """
    Get the current global tropical mode.

    :return: The current TropicalMode.
    """
    return _global_tropical_mode


@contextmanager
def tropical_mode(mode: TropicalMode):
    """
    Context manager to temporarily set the tropical mode.

    :param mode: The TropicalMode to set for the duration of the context.
    """
    global _global_tropical_mode
    original_mode = _global_tropical_mode
    try:
        set_global_tropical_mode(mode)
        yield
    finally:
        set_global_tropical_mode(original_mode)
