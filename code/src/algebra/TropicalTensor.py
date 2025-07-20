from __future__ import annotations
import numpy as np
from typing import Any, Optional
from .TropicalGlobal import get_global_tropical_mode
from .TropicalElement import TropicalElement
from .types import TropicalMode


class TropicalTensor(np.ndarray):
    """
    A class representing a tensor of TropicalElements.
    Inherits from NumPy's ndarray to leverage its functionality.
    """

    _mode: TropicalMode | None = None

    def __new__(
        cls, input_array: np.ndarray | list, mode: TropicalMode | None = None
    ) -> TropicalTensor:
        """
        Create a new TropicalTensor instance.
        :param input_array: The input array containing TropicalElements.
        :param mode: Enforce all elements to have the same tropical mode.
        """
        base = np.asarray(input_array)
        flat_arr = base.flatten()
        transformed_arr = []
        for item in flat_arr:
            if isinstance(item, TropicalElement):
                if (mode is not None and item._mode == mode) or (
                    mode is None and item._mode is not None
                ):
                    raise ValueError(
                        f"All elements must have the same mode: {mode}, but found {item._mode}."
                    )
                transformed_arr.append(item)
            else:
                transformed_arr.append(TropicalElement(item, mode=mode))

        wrapped_arr = np.array(transformed_arr, dtype=object).reshape(base.shape)
        obj = wrapped_arr.view(cls)
        obj._mode = mode
        return obj

    @property
    def mode(self) -> TropicalMode:
        """
        Get the TropicalMode of the matrix.
        """
        return self._mode if self._mode is not None else get_global_tropical_mode()

    def __array_finalize__(self, parent: Optional[np.ndarray]) -> None:
        if parent is None:
            return
        self._mode = getattr(parent, "_mode", None)

    def __array_function__(
        self,
        func: Any,
        types: tuple[type, ...],
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        # Intercept only np.linalg.det
        # TODO: See if we really need this functionality
        if func is np.linalg.det:
            raise NotImplementedError(
                "Determinant calculation is not supported for TropicalMatrix."
            )
        # Fallback to default behavior
        # pylint: disable=no-member
        return np.ndarray.__array_function__(self, func, types, args, kwargs)
