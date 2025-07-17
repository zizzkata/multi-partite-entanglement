from __future__ import annotations
from typing import TypeAlias
from enum import Enum, auto
import math
from contextlib import contextmanager


class TropicalMode(Enum):
    """
    Enum representing the modes of tropical algebra.
    """

    MIN_PLUS = auto()
    MAX_PLUS = auto()


_global_tropical_mode = TropicalMode.MIN_PLUS


def set_global_tropical_mode(mode: TropicalMode):
    """
    Set the global tropical mode for the algebra operations.

    :param mode: The TropicalMode to set as global.
    """
    global _global_tropical_mode
    if mode not in [TropicalMode.MIN_PLUS, TropicalMode.MAX_PLUS]:
        raise ValueError(
            "Invalid tropical mode. Use TropicalMode.MIN_PLUS or TropicalMode.MAX_PLUS."
        )
    _global_tropical_mode = mode


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


class TropicalElement:
    """
    A class representing an element in the tropical algebra.
    Or more specifically, an element in the tropical semiring(min/max)-plus.
    """

    _mode: TropicalMode | None

    _MULTIPLICATIVE_IDENTITY: float = 0

    @classmethod
    def additive_identity(cls) -> TropicalElement:
        """
        Create a zero tropical element
        based on the current mode or the envireonment.
        """
        mode = (
            _global_tropical_mode if not isinstance(cls, TropicalElement) else cls.mode
        )
        return cls(math.inf if mode == TropicalMode.MIN_PLUS else -math.inf)

    @classmethod
    def multiplicative_identity(cls) -> TropicalElement:
        """
        Create a one tropical element based on the current mode.
        """
        return cls(cls._MULTIPLICATIVE_IDENTITY)

    def _require_compatible(self, other: TropicalScalar):
        """
        Ensure that the other tropical element is compatible with this one.

        :param other: Another TropicalElement to check compatibility with.
        :raises ValueError: If the modes of the two elements are not compatible.
        """
        if not isinstance(other, (TropicalElement, int, float)):
            raise TypeError(
                "Can only operate with TropicalElement | int | float instances."
            )
        elif isinstance(other, (int, float)):
            other = TropicalElement(other, mode=self.mode)
        if self.mode != other.mode:
            raise ValueError(
                "Tropical elements must have the same mode for this operation."
            )

        if isinstance(other, TropicalElement):
            return other
        return TropicalElement(other, mode=self.mode)

    def __init__(self, value, mode: TropicalMode | None = None):
        """
        Initialize a TropicalElement with a value and an optional mode.

        :param value: The value of the tropical element.
        :param mode: The operation mode, either 'min' or 'max'.
        """
        self.value = value
        if mode not in [TropicalMode.MAX_PLUS, TropicalMode.MIN_PLUS, None]:
            raise ValueError("Mode must be either 'min' or 'max'.")
        self._mode = mode

    @property
    def mode(self) -> TropicalMode:
        """
        Get the current mode of the tropical element.

        :return: The TropicalMode of the element.
        """
        return self._mode if self._mode is not None else _global_tropical_mode

    def __add__(self, other: TropicalScalar) -> TropicalElement:
        """
        Define addition for tropical elements based on the mode.

        :param other: Another TropicalElement to add.
        :return: A new TropicalElement representing the result of the addition.
        """
        other = self._require_compatible(other)
        fn_operation = min if self.mode == TropicalMode.MIN_PLUS else max
        return type(self)(fn_operation(self.value, other.value), self.mode)

    def __radd__(self, other: TropicalScalar) -> TropicalElement:
        """
        Define right addition for tropical elements based on the mode.

        :param other: Another TropicalElement to add.
        :return: A new TropicalElement representing the result of the right addition.
        """
        return self.__add__(other)

    def __neg__(self) -> TropicalElement:
        """
        Negate the tropical element.
        """
        return type(self)(-self.value, self.mode)

    def __sub__(self, other: TropicalScalar) -> TropicalElement:
        """
        Define subtraction for tropical elements based on the mode.
        This is defined as addition with the additive inverse.

        :param other: Another TropicalElement to subtract.
        :return: A new TropicalElement representing the result of the subtraction.
        """
        other = self._require_compatible(other)
        other = -other  # Negate the value for subtraction
        return self.__add__(other)

    def __rsub__(self, other: TropicalScalar) -> TropicalElement:
        """
        Define right subtraction for tropical elements based on the mode.
        This is defined as addition with the additive inverse.

        :param other: Another TropicalElement to subtract from.
        :return: A new TropicalElement representing the result of the right subtraction.
        """
        # other - self = -1 (self - other)
        res = -self.__sub__(other)
        return res

    def __mul__(self, other: TropicalScalar) -> TropicalElement:
        """
        Define multiplication for tropical elements based on the mode.
        """
        other = self._require_compatible(other)
        return type(self)(self.value + other.value, self.mode)

    def __rmul__(self, other: TropicalScalar) -> TropicalElement:
        """
        Define right multiplication for tropical elements based on the mode.
        """
        return self.__mul__(other)

    def __truediv__(self, other: TropicalScalar) -> TropicalElement:
        """
        Define division for tropical elements based on the mode.
        This is defined as multiplication with the multiplicative inverse.
        """
        other = self._require_compatible(other)
        other = -other  # Negate the value for division (multiplicative inverse)
        return self.__mul__(other)

    def __rtruediv__(self, other: TropicalScalar) -> TropicalElement:
        """
        Define right division for tropical elements based on the mode.
        This is defined as multiplication with the multiplicative inverse.
        """
        self_val = -self  # Negate the value for right division
        return self_val.__mul__(other)

    def __pow__(self, exponent) -> TropicalElement:
        """
        Define exponentiation for tropical elements based on the mode.
        This is defined as repeated multiplication.

        :param exponent: The exponent to raise the tropical element to.
        :return: A new TropicalElement representing the result of the exponentiation.
        """
        return type(self)(self.value * exponent, self.mode)

    def __repr__(self):
        """
        String representation of the element.
        """
        return str(self.value)

    def __eq__(self, other: TropicalScalar) -> bool:
        """
        Define equality for tropical elements.
        """
        other = self._require_compatible(other)
        return self.value == other.value

    def is_same_object(self, other: TropicalScalar) -> bool:
        """
        Check if two tropical elements are the same object.
        """
        other = self._require_compatible(other)
        return self is other

    def __array__(self, dtype=None):
        raise TypeError("Cannot convert Tropical to a raw array")


TropicalScalar: TypeAlias = TropicalElement | int | float
