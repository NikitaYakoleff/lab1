# cython: language_level=3, boundscheck=False, wraparound=False, nonecheck=False


def sum_of_squares_cy(double[:] arr):
    """Сумма квадратов элементов массива (типизированный C-цикл)."""
    cdef Py_ssize_t i
    cdef Py_ssize_t n = arr.shape[0]
    cdef double total = 0.0
    for i in range(n):
        total += arr[i] * arr[i]
    return total
