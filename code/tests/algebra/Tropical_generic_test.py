import pytest
import math
from algebra import (
    TropicalElement,
    TropicalMode,
    tropical_mode,
    set_global_tropical_mode,
)


def test_tropical_element_initialization():
    """
    Test the initialization of a TropicalElement with a value and default mode.
    """
    element = TropicalElement(5)
    assert element.value == 5
    # default mode is MIN_PLUS
    assert element.mode == TropicalMode.MIN_PLUS
    assert element._mode is None


def test_tropical_element_with_mode():
    """
    Test the initialization of a TropicalElement with a value and a specific mode.
    """
    element = TropicalElement(3, mode=TropicalMode.MAX_PLUS)
    assert element.value == 3
    assert element.mode == TropicalMode.MAX_PLUS
    assert element._mode == TropicalMode.MAX_PLUS


def test_tropical_element_mode_context():
    """
    Test the scope of a TropicalElement.
    """
    with tropical_mode(TropicalMode.MIN_PLUS):
        element = TropicalElement(10)
        assert element.mode == TropicalMode.MIN_PLUS

    with tropical_mode(TropicalMode.MAX_PLUS):
        element = TropicalElement(20)
        assert element.mode == TropicalMode.MAX_PLUS


def test_tropical_element_mode_context_limits():
    """
    Test that the tropical mode context manager correctly sets and resets the mode.
    """
    a = TropicalElement(1)
    b = TropicalElement(2)
    c = TropicalElement(3, mode=TropicalMode.MAX_PLUS)

    with tropical_mode(TropicalMode.MIN_PLUS):
        assert a.mode == TropicalMode.MIN_PLUS
        assert b.mode == TropicalMode.MIN_PLUS
        assert c.mode == TropicalMode.MAX_PLUS

    with tropical_mode(TropicalMode.MAX_PLUS):
        assert a.mode == TropicalMode.MAX_PLUS
        assert b.mode == TropicalMode.MAX_PLUS
        assert c.mode == TropicalMode.MAX_PLUS


def test_tropical_element_global_variable():
    """
    Test that the global variable _mode is set correctly when using tropical_mode.
    """
    try:
        a = TropicalElement(1)
        b = TropicalElement(2)
        assert a.mode == TropicalMode.MIN_PLUS
        assert b.mode == TropicalMode.MIN_PLUS

        set_global_tropical_mode(TropicalMode.MAX_PLUS)
        c = TropicalElement(3)
        assert c.mode == TropicalMode.MAX_PLUS
        assert a.mode == TropicalMode.MAX_PLUS
        assert b.mode == TropicalMode.MAX_PLUS
    finally:
        # Reset global mode to default after test
        set_global_tropical_mode(TropicalMode.MIN_PLUS)


def test_tropical_element_plus():
    """
    Test the addition operation for TropicalElement.
    """
    a = TropicalElement(1)
    b = TropicalElement(2)

    with tropical_mode(TropicalMode.MIN_PLUS):
        result = a + b
        assert result.value == 1

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = a + b
        assert result.value == 2


def test_tropical_element_multiply():
    """
    Test the multiplication operation for TropicalElement.
    """
    a = TropicalElement(3)
    b = TropicalElement(4)

    with tropical_mode(TropicalMode.MIN_PLUS):
        result = a * b
        assert result.value == 7

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = a * b
        assert result.value == 7


def test_tropical_element_equality():
    """
    Test the equality operation for TropicalElement.
    """
    a = TropicalElement(5)
    b = TropicalElement(5)
    c = TropicalElement(6)

    assert a == b
    assert a != c
    assert b != c

    with tropical_mode(TropicalMode.MAX_PLUS):
        d = TropicalElement(5, mode=TropicalMode.MAX_PLUS)
        assert a == d
        assert b == d
        assert c != d


def test_tropical_element_is_same_object():
    """
    Test the is_same_object method for TropicalElement.
    """
    a = TropicalElement(5)
    b = TropicalElement(5)
    c = a

    assert a.is_same_object(b) is False
    assert a.is_same_object(c) is True

    with tropical_mode(TropicalMode.MAX_PLUS):
        d = TropicalElement(5, mode=TropicalMode.MAX_PLUS)
        assert a.is_same_object(d) is False
        assert c.is_same_object(d) is False


def test_tropical_element_identities():
    """
    Test the identities of TropicalElement.
    """
    a = TropicalElement(2)
    with tropical_mode(TropicalMode.MIN_PLUS):
        assert a.additive_identity() == math.inf
        assert 0 == a.multiplicative_identity()
        assert TropicalElement.additive_identity() == math.inf
        assert TropicalElement.multiplicative_identity() == 0

    with tropical_mode(TropicalMode.MAX_PLUS):
        assert a.additive_identity() == -math.inf
        assert a.multiplicative_identity() == 0
        assert TropicalElement.additive_identity() == -math.inf
        assert TropicalElement.multiplicative_identity() == 0


def test_tropical_additive_with_negative_element():
    """
    Test the addition of TropicalElement with a negative element.
    """
    a = TropicalElement(5)
    with tropical_mode(TropicalMode.MIN_PLUS):
        assert a + (-3) == -3
        assert (-3) + a == -3
        assert a - 3 == -3
        assert -3 + a == -3
        assert a * (-3) == 2
        assert (-3) * a == 2

    with tropical_mode(TropicalMode.MAX_PLUS):
        assert a + (-3) == 5
        assert (-3) + a == 5
        assert a - 3 == 5
        assert -3 + a == 5
        assert a * (-3) == 2
        assert (-3) * a == 2


def test_tropical_multiplicative_inverse():
    """
    Test the multplicative inverse of TropicalElement.
    """
    a = TropicalElement(5)
    with tropical_mode(TropicalMode.MIN_PLUS):
        assert a * (-5) == TropicalElement.multiplicative_identity()
        assert (
            a * (TropicalElement.multiplicative_identity() / a)
            == TropicalElement.multiplicative_identity()
        )
        assert (-5) * a == TropicalElement.multiplicative_identity()
        assert (
            TropicalElement.multiplicative_identity() / a
        ) * a == TropicalElement.multiplicative_identity()

    with tropical_mode(TropicalMode.MAX_PLUS):
        assert a * (-5) == TropicalElement.multiplicative_identity()
        assert (
            a * (TropicalElement.multiplicative_identity() / a)
            == TropicalElement.multiplicative_identity()
        )
        assert (-5) * a == TropicalElement.multiplicative_identity()
        assert (
            TropicalElement.multiplicative_identity() / a
        ) * a == TropicalElement.multiplicative_identity()


def test_tropical_element_power():
    """
    Test the power operation for TropicalElement.
    """
    a = TropicalElement(2)
    with tropical_mode(TropicalMode.MIN_PLUS):
        result = a**3
        assert result.value == 6  # 2 + 2 + 2
        result = a**0
        assert (
            result.value == TropicalElement.multiplicative_identity()
        )  # 0 in MIN_PLUS mode
        result = a**1
        assert result.value == 2  # 2^1 = 2
        result = a**-1
        assert result.value == -2  # 2^-1 = -2
        assert (a**-1) * a == TropicalElement.multiplicative_identity()  # 2^-1 * 2 = 0
        # test sqrt2 a**n = a*n
        sqrt2 = a**0.5
        assert sqrt2.value == 1

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = a**3
        assert result.value == 6  # 2 + 2 + 2
        result = a**0
        assert result.value == TropicalElement.multiplicative_identity()  # 0 in MAX
        result = a**1
        assert result.value == 2  # 2^1 = 2
        result = a**-1
        assert result.value == -2  # 2^-1 = -2
        assert (a**-1) * a == TropicalElement.multiplicative_identity()
        # test sqrt2 a**n = a*n
        sqrt2 = a**0.5
        assert sqrt2.value == 1  # 2^0.5 = 1
