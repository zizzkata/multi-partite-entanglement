import pytest
import numpy as np
from algebra import TropicalElement, tropical_mode, TropicalMode, TropicalTensor


def test_numpy_array_initialization():
    """
    Test the initialization of a TropicalElement with a NumPy array.
    """
    array = np.array([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    assert isinstance(array, np.ndarray)
    assert array.shape == (3,)
    array2 = np.array(
        [[TropicalElement(4)], [TropicalElement(5)], [TropicalElement(6)]]
    )
    assert isinstance(array2, np.ndarray)
    assert array2.shape == (3, 1)


def test_tropical_tensor_initialization():
    """
    Test the initialization of a TropicalTensor with a NumPy array.
    """
    matrix = TropicalTensor([1, 2, 3])
    assert isinstance(matrix, TropicalTensor)
    assert matrix.shape == (3,)
    matrix2 = TropicalTensor([[1], [2], [3]])
    assert isinstance(matrix2, TropicalTensor)
    assert matrix2.shape == (3, 1)


def test_numpy_array_addition():
    """
    Test operations on NumPy arrays containing TropicalElements.
    """
    array1 = np.array([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    array2 = np.array([TropicalElement(4), TropicalElement(5), TropicalElement(6)])
    with tropical_mode(TropicalMode.MIN_PLUS):
        result = array1 + array2
        assert result[0].value == 1
        assert result[1].value == 2
        assert result[2].value == 3

        result = array1 - array2
        assert result[0].value == -4
        assert result[1].value == -5
        assert result[2].value == -6

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = array1 + array2
        assert result[0].value == 4
        assert result[1].value == 5
        assert result[2].value == 6
        result = array1 - array2
        assert result[0].value == 1
        assert result[1].value == 2
        assert result[2].value == 3


def test_tropical_tensor_addition():
    """
    Test operations on TropicalTensor objects.
    """
    matrix1 = TropicalTensor([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    matrix2 = TropicalTensor([TropicalElement(4), TropicalElement(5), TropicalElement(6)])
    with tropical_mode(TropicalMode.MIN_PLUS):
        result = matrix1 + matrix2
        assert result[0].value == 1
        assert result[1].value == 2
        assert result[2].value == 3

        result = matrix1 - matrix2
        assert result[0].value == -4
        assert result[1].value == -5
        assert result[2].value == -6

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = matrix1 + matrix2
        assert result[0].value == 4
        assert result[1].value == 5
        assert result[2].value == 6
        result = matrix1 - matrix2
        assert result[0].value == 1
        assert result[1].value == 2
        assert result[2].value == 3


def test_numpy_array_output():
    """
    Test the output of NumPy arrays containing TropicalElements.
    """
    array = np.array([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    array_normal = np.array([1, 2, 3])
    with tropical_mode(TropicalMode.MIN_PLUS):
        res = array + array_normal
        assert isinstance(res[0], TropicalElement)
        assert res[0].value == 1
        assert isinstance(res[1], TropicalElement)
        assert res[1].value == 2
        assert isinstance(res[2], TropicalElement)
        assert res[2].value == 3

    with tropical_mode(TropicalMode.MAX_PLUS):
        res = array + array_normal
        assert isinstance(res[0], TropicalElement)
        assert res[0].value == 1
        assert isinstance(res[1], TropicalElement)
        assert res[1].value == 2
        assert isinstance(res[2], TropicalElement)
        assert res[2].value == 3


def test_tropical_tensor_output():
    """
    Test the output of TropicalTensor objects.
    """
    matrix = TropicalTensor([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    matrix_normal = TropicalTensor([1, 2, 3])
    with tropical_mode(TropicalMode.MIN_PLUS):
        res = matrix + matrix_normal
        assert isinstance(res[0], TropicalElement)
        assert res[0].value == 1
        assert isinstance(res[1], TropicalElement)
        assert res[1].value == 2
        assert isinstance(res[2], TropicalElement)
        assert res[2].value == 3

    with tropical_mode(TropicalMode.MAX_PLUS):
        res = matrix + matrix_normal
        assert isinstance(res[0], TropicalElement)
        assert res[0].value == 1
        assert isinstance(res[1], TropicalElement)
        assert res[1].value == 2
        assert isinstance(res[2], TropicalElement)
        assert res[2].value == 3


def test_numpy_array_multiplication_normal():
    """
    Test multiplication of NumPy arrays containing TropicalElements.
    """
    array1 = np.array([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    array2 = np.array([TropicalElement(4), TropicalElement(5), TropicalElement(6)])

    with tropical_mode(TropicalMode.MIN_PLUS):
        result = array1 * array2
        assert result[0].value == 5
        assert result[1].value == 7
        assert result[2].value == 9

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = array1 * array2
        assert result[0].value == 5
        assert result[1].value == 7
        assert result[2].value == 9


def test_tropical_tensor_multiplication_normal():
    """
    Test multiplication of TropicalTensor objects.
    """
    matrix1 = TropicalTensor([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    matrix2 = TropicalTensor([TropicalElement(4), TropicalElement(5), TropicalElement(6)])

    with tropical_mode(TropicalMode.MIN_PLUS):
        result = matrix1 * matrix2
        assert result[0].value == 5
        assert result[1].value == 7
        assert result[2].value == 9

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = matrix1 * matrix2
        assert result[0].value == 5
        assert result[1].value == 7
        assert result[2].value == 9


def test_numpy_array_multiplication_with_scalar():
    """
    Test multiplication of a NumPy array containing TropicalElements with a scalar.
    """
    array = np.array([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    scalar = 2

    with tropical_mode(TropicalMode.MIN_PLUS):
        result = array * scalar
        assert result[0].value == 3
        assert result[1].value == 4
        assert result[2].value == 5

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = array * scalar
        assert result[0].value == 3
        assert result[1].value == 4
        assert result[2].value == 5


def test_tropical_tensor_multiplication_with_scalar():
    """
    Test multiplication of a TropicalTensor with a scalar.
    """
    matrix = TropicalTensor([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    scalar = 2

    with tropical_mode(TropicalMode.MIN_PLUS):
        result = matrix * scalar
        assert result[0].value == 3
        assert result[1].value == 4
        assert result[2].value == 5

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = matrix * scalar
        assert result[0].value == 3
        assert result[1].value == 4
        assert result[2].value == 5


def test_numpy_array_equality():
    """
    Test equality of NumPy arrays containing TropicalElements.
    """
    array = np.array([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    array2 = np.array([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    array3 = np.array([TropicalElement(4), TropicalElement(5), TropicalElement(6)])
    assert np.array_equal(array, array2)
    assert not np.array_equal(array, array3)
    assert not np.array_equal(array2, array3)
    assert array is not array2  # Different objects, same values


def test_tropical_tensor_equality():
    """
    Test equality of TropicalTensor objects.
    """
    matrix = TropicalTensor([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    matrix2 = TropicalTensor([TropicalElement(1), TropicalElement(2), TropicalElement(3)])
    matrix3 = TropicalTensor([TropicalElement(4), TropicalElement(5), TropicalElement(6)])
    assert np.array_equal(matrix, matrix2)
    assert not np.array_equal(matrix, matrix3)
    assert not np.array_equal(matrix2, matrix3)
    assert matrix is not matrix2  # Different objects, same values


def test_numpy_array_matrix_multiplication1():
    """
    Test matrix multiplication of NumPy arrays containing TropicalElements.
    """
    array1 = np.array(
        [
            [TropicalElement(1)],
            [TropicalElement(2)],
            [TropicalElement(3)],
        ]
    )
    array2 = np.array(
        [
            [TropicalElement(4)],
            [TropicalElement(5)],
            [TropicalElement(6)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        result = array1 @ array2.T  # Matrix multiplication
        assert result[0][0].value == 5
        assert result[0][1].value == 6
        assert result[0][2].value == 7
        assert result[1][0].value == 6
        assert result[1][1].value == 7
        assert result[1][2].value == 8
        assert result[2][0].value == 7
        assert result[2][1].value == 8
        assert result[2][2].value == 9

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = array1 @ array2.T  # Matrix multiplication
        assert result[0][0].value == 5
        assert result[0][1].value == 6
        assert result[0][2].value == 7
        assert result[1][0].value == 6
        assert result[1][1].value == 7
        assert result[1][2].value == 8
        assert result[2][0].value == 7
        assert result[2][1].value == 8
        assert result[2][2].value == 9


def test_tropical_tensor_matrix_multiplication1():
    """
    Test matrix multiplication of TropicalTensor objects.
    """
    matrix1 = TropicalTensor(
        [
            [TropicalElement(1)],
            [TropicalElement(2)],
            [TropicalElement(3)],
        ]
    )
    matrix2 = TropicalTensor(
        [
            [TropicalElement(4)],
            [TropicalElement(5)],
            [TropicalElement(6)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        result = matrix1 @ matrix2.T  # Matrix multiplication
        assert result[0][0].value == 5
        assert result[0][1].value == 6
        assert result[0][2].value == 7
        assert result[1][0].value == 6
        assert result[1][1].value == 7
        assert result[1][2].value == 8
        assert result[2][0].value == 7
        assert result[2][1].value == 8
        assert result[2][2].value == 9

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = matrix1 @ matrix2.T  # Matrix multiplication
        assert result[0][0].value == 5
        assert result[0][1].value == 6
        assert result[0][2].value == 7
        assert result[1][0].value == 6
        assert result[1][1].value == 7
        assert result[1][2].value == 8
        assert result[2][0].value == 7
        assert result[2][1].value == 8
        assert result[2][2].value == 9


def test_numpy_array_matrix_multiplication2():
    """
    Test matrix multiplication of NumPy arrays containing TropicalElements.
    """
    array1 = np.array(
        [
            [TropicalElement(1), TropicalElement(2)],
            [TropicalElement(3), TropicalElement(4)],
        ]
    )
    array2 = np.array(
        [
            [TropicalElement(4), TropicalElement(5)],
            [TropicalElement(6), TropicalElement(7)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        result = array1 @ array2  # Matrix multiplication
        assert result[0][0].value == 5
        assert result[0][1].value == 6
        assert result[1][0].value == 7
        assert result[1][1].value == 8

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = array1 @ array2  # Matrix multiplication
        assert result[0][0].value == 8
        assert result[0][1].value == 9
        assert result[1][0].value == 10
        assert result[1][1].value == 11


def test_tropical_tensor_matrix_multiplication2():
    """
    Test matrix multiplication of TropicalTensor objects.
    """
    matrix1 = TropicalTensor(
        [
            [TropicalElement(1), TropicalElement(2)],
            [TropicalElement(3), TropicalElement(4)],
        ]
    )
    matrix2 = TropicalTensor(
        [
            [TropicalElement(4), TropicalElement(5)],
            [TropicalElement(6), TropicalElement(7)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        result = matrix1 @ matrix2  # Matrix multiplication
        assert result[0][0].value == 5
        assert result[0][1].value == 6
        assert result[1][0].value == 7
        assert result[1][1].value == 8

    with tropical_mode(TropicalMode.MAX_PLUS):
        result = matrix1 @ matrix2  # Matrix multiplication
        assert result[0][0].value == 8
        assert result[0][1].value == 9
        assert result[1][0].value == 10
        assert result[1][1].value == 11


def test_numpy_array_matrix_trace():
    """
    Test the trace of a NumPy array containing TropicalElements.
    """
    array = np.array(
        [
            [TropicalElement(1), TropicalElement(2)],
            [TropicalElement(3), TropicalElement(4)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        trace = np.trace(array)
        assert trace.value == 1

    with tropical_mode(TropicalMode.MAX_PLUS):
        trace = np.trace(array)
        assert trace.value == 4


def test_tropical_tensor_trace():
    """
    Test the trace of a TropicalTensor.
    """
    matrix = TropicalTensor(
        [
            [TropicalElement(1), TropicalElement(2)],
            [TropicalElement(3), TropicalElement(4)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        trace = np.trace(matrix)
        assert trace.value == 1

    with tropical_mode(TropicalMode.MAX_PLUS):
        trace = np.trace(matrix)
        assert trace.value == 4


def test_numpy_array_matrix_transpose():
    """
    Test the transpose of a NumPy array containing TropicalElements.
    """
    array = np.array(
        [
            [TropicalElement(1), TropicalElement(2)],
            [TropicalElement(3), TropicalElement(4)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        transposed = array.T
        assert transposed[0][0].value == 1
        assert transposed[0][1].value == 3
        assert transposed[1][0].value == 2
        assert transposed[1][1].value == 4
    with tropical_mode(TropicalMode.MAX_PLUS):
        transposed = array.T
        assert transposed[0][0].value == 1
        assert transposed[0][1].value == 3
        assert transposed[1][0].value == 2
        assert transposed[1][1].value == 4


def test_tropical_tensor_transpose():
    """
    Test the transpose of a TropicalTensor.
    """
    matrix = TropicalTensor(
        [
            [TropicalElement(1), TropicalElement(2)],
            [TropicalElement(3), TropicalElement(4)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        transposed = matrix.T
        assert transposed[0][0].value == 1
        assert transposed[0][1].value == 3
        assert transposed[1][0].value == 2
        assert transposed[1][1].value == 4
    with tropical_mode(TropicalMode.MAX_PLUS):
        transposed = matrix.T
        assert transposed[0][0].value == 1
        assert transposed[0][1].value == 3
        assert transposed[1][0].value == 2
        assert transposed[1][1].value == 4


def test_numpy_array_matrix_power():
    """
    Test the power of a NumPy array containing TropicalElements.
    """
    array = np.array(
        [
            [TropicalElement(1), TropicalElement(2)],
            [TropicalElement(3), TropicalElement(4)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        powered = np.linalg.matrix_power(array, 2)
        assert powered[0][0].value == 2
        assert powered[0][1].value == 3
        assert powered[1][0].value == 4
        assert powered[1][1].value == 5

    with tropical_mode(TropicalMode.MAX_PLUS):
        powered = np.linalg.matrix_power(array, 2)
        assert powered[0][0].value == 5
        assert powered[0][1].value == 6
        assert powered[1][0].value == 7
        assert powered[1][1].value == 8


def test_tropical_tensor_power():
    """
    Test the power of a TropicalTensor.
    """
    matrix = TropicalTensor(
        [
            [TropicalElement(1), TropicalElement(2)],
            [TropicalElement(3), TropicalElement(4)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        powered = np.linalg.matrix_power(matrix, 2)
        assert powered[0][0].value == 2
        assert powered[0][1].value == 3
        assert powered[1][0].value == 4
        assert powered[1][1].value == 5

    with tropical_mode(TropicalMode.MAX_PLUS):
        powered = np.linalg.matrix_power(matrix, 2)
        assert powered[0][0].value == 5
        assert powered[0][1].value == 6
        assert powered[1][0].value == 7
        assert powered[1][1].value == 8


def test_numpy_array_tropical_kron_prod():
    """
    Test the Kronecker product of TropicalMatrices.
    """
    matrix1 = np.array(
        [
            [TropicalElement(1), TropicalElement(2)],
            [TropicalElement(3), TropicalElement(4)],
        ]
    )
    matrix2 = np.array(
        [
            [TropicalElement(5), TropicalElement(6)],
            [TropicalElement(7), TropicalElement(8)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        kron_prod = np.kron(matrix1, matrix2)
        assert kron_prod.shape == (4, 4)
        for matrix1_i in range(4):
            i1, i2 = matrix1_i // 2, matrix1_i % 2
            for matrix2_j in range(4):
                j1, j2 = matrix2_j // 2, matrix2_j % 2
                kron_i1, kron_i2 = i1 * 2 + j1, i2 * 2 + j2
                assert kron_prod[kron_i1][kron_i2].value == (
                    matrix1[i1][i2].value + matrix2[j1][j2].value
                )
    with tropical_mode(TropicalMode.MAX_PLUS):
        kron_prod = np.kron(matrix1, matrix2)
        assert kron_prod.shape == (4, 4)
        for matrix1_i in range(4):
            i1, i2 = matrix1_i // 2, matrix1_i % 2
            for matrix2_j in range(4):
                j1, j2 = matrix2_j // 2, matrix2_j % 2
                kron_i1, kron_i2 = i1 * 2 + j1, i2 * 2 + j2
                assert kron_prod[kron_i1][kron_i2].value == (
                    matrix1[i1][i2].value + matrix2[j1][j2].value
                )


def test_tropical_tensor_tropical_kron_prod():
    """
    Test the Kronecker product of TropicalMatrices.
    """
    matrix1 = TropicalTensor(
        [
            [TropicalElement(1), TropicalElement(2)],
            [TropicalElement(3), TropicalElement(4)],
        ]
    )
    matrix2 = TropicalTensor(
        [
            [TropicalElement(5), TropicalElement(6)],
            [TropicalElement(7), TropicalElement(8)],
        ]
    )

    with tropical_mode(TropicalMode.MIN_PLUS):
        kron_prod = np.kron(matrix1, matrix2)
        assert kron_prod.shape == (4, 4)
        for matrix1_i in range(4):
            i1, i2 = matrix1_i // 2, matrix1_i % 2
            for matrix2_j in range(4):
                j1, j2 = matrix2_j // 2, matrix2_j % 2
                kron_i1, kron_i2 = i1 * 2 + j1, i2 * 2 + j2
                assert kron_prod[kron_i1][kron_i2].value == (
                    matrix1[i1][i2].value + matrix2[j1][j2].value
                )
    with tropical_mode(TropicalMode.MAX_PLUS):
        kron_prod = np.kron(matrix1, matrix2)
        assert kron_prod.shape == (4, 4)
        for matrix1_i in range(4):
            i1, i2 = matrix1_i // 2, matrix1_i % 2
            for matrix2_j in range(4):
                j1, j2 = matrix2_j // 2, matrix2_j % 2
                kron_i1, kron_i2 = i1 * 2 + j1, i2 * 2 + j2
                assert kron_prod[kron_i1][kron_i2].value == (
                    matrix1[i1][i2].value + matrix2[j1][j2].value
                )


def test_tropical_tensor_contraction():
    """
    Test the contraction of a TropicalTensor.
    """
    m1 = np.array([
        [TropicalElement(1), TropicalElement(2)],
        [TropicalElement(3), TropicalElement(4)],
    ])
    m2 = np.array([
        [TropicalElement(5), TropicalElement(6)],
        [TropicalElement(7), TropicalElement(8)],
    ])
    m3 = np.array([
        [TropicalElement(9), TropicalElement(10)],
        [TropicalElement(11), TropicalElement(12)],
    ])
    m4 = np.array([
        [TropicalElement(13), TropicalElement(14)],
        [TropicalElement(15), TropicalElement(16)],
    ])
    tensor1 = np.kron(m1, m2)
    tensor2 = np.kron(m3, m4)

    with tropical_mode(TropicalMode.MIN_PLUS):
        # contracted = tensor.contract()
        # assert contracted.value == 1
        pass

    with tropical_mode(TropicalMode.MAX_PLUS):
        # contracted = tensor.contract()
        # assert contracted.value == 4
        pass
