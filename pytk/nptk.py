import numpy as np

from more_itertools.recipes import pairwise


def split(a, slices):
    # TODO raise exception is slices don't match size of a
    cumslices = np.insert(np.cumsum(slices), 0, 0)
    return tuple(a[sf:st] for sf, st in pairwise(cumslices))


# TODO probably exists np.apply_along_axis
def do_along(f, M, axis):
    M2 = np.swapaxes(M, 0, axis)
    ndim = M2.shape[0]
    return np.array([f(M2[i, ...]) for i in range(ndim)])


# if __name__ == '__main__':
#     a = np.array([1, 2, 3, 4, 5])
#     b = np.array([2, 3])
#     c = np.array([3, 2])
#     d = np.array([2, 1, 2])
#     print(slice_as(a, b))
#     print(slice_as(a, c))
#     print(slice_as(a, d))

###############################################################################


# def _assertRank2(*arrays):
#     for a in arrays:
#         if len(a.shape) != 2:
#             raise LinAlgError('%d-dimensional array given. Array must be '
#                               'two-dimensional' % len(a.shape))


###############################################################################
# multi_dot
# def multi_dot(arrays):
#     """
#     Compute the dot product of two or more arrays in a single function call,
#     while automatically selecting the fastest evaluation order.

#     `multi_dot` chains `numpy.dot` and uses optimal parenthesization
#     of the matrices [1]_ [2]_. Depending on the shapes of the matrices,
#     this can speed up the multiplication a lot.

#     If the first argument is 1-D it is treated as a row vector.
#     If the last argument is 1-D it is treated as a column vector.
#     The other arguments must be 2-D.

#     Think of `multi_dot` as::

#         def multi_dot(arrays): return functools.reduce(np.dot, arrays)


#     Parameters
#     ----------
#     arrays : sequence of array_like
#         If the first argument is 1-D it is treated as row vector.
#         If the last argument is 1-D it is treated as column vector.
#         The other arguments must be 2-D.

#     Returns
#     -------
#     output : ndarray
#         Returns the dot product of the supplied arrays.

#     See Also
#     --------
#     dot : dot multiplication with two arguments.

#     References
#     ----------

#     .. [1] Cormen, "Introduction to Algorithms", Chapter 15.2, p. 370-378
#     .. [2] http://en.wikipedia.org/wiki/Matrix_chain_multiplication

#     Examples
#     --------
#     `multi_dot` allows you to write::

#     >>> from nphelper import multi_dot
#     >>> # Prepare some data
#     >>> A = np.random.random((10000, 100))
#     >>> B = np.random.random((100, 1000))
#     >>> C = np.random.random((1000, 5))
#     >>> D = np.random.random((5, 333))
#     >>> # the actual dot multiplication
#     >>> multi_dot([A, B, C, D])  # doctest: +SKIP

#     instead of::

#     >>> np.dot(np.dot(np.dot(A, B), C), D)  # doctest: +SKIP
#     >>> # or
#     >>> A.dot(B).dot(C).dot(D)  # doctest: +SKIP


#     Example: multiplication costs of different parenthesizations
#     ------------------------------------------------------------

#     The cost for a matrix multiplication can be calculated with the
#     following function::

#         def cost(A, B): return A.shape[0] * A.shape[1] * B.shape[1]

#     Let's assume we have three matrices
#     :math:`A_{10x100}, B_{100x5}, C_{5x50}$`.

#     The costs for the two different parenthesizations are as follows::

#         cost((AB)C) = 10*100*5 + 10*5*50   = 5000 + 2500   = 7500
#         cost(A(BC)) = 10*100*50 + 100*5*50 = 50000 + 25000 = 75000

#     """
#     n = len(arrays)
#     # optimization only makes sense for len(arrays) > 2
#     if n < 2:
#         raise ValueError("Expecting at least two arrays.")
#     elif n == 2:
#         return np.dot(arrays[0], arrays[1])

#     arrays = [np.asanyarray(a) for a in arrays]

#     # save original ndim to reshape the result array into the proper form later
#     ndim_first, ndim_last = arrays[0].ndim, arrays[-1].ndim
#     # Explicitly convert vectors to 2D arrays to keep the logic of the internal
#     # _multi_dot_* functions as simple as possible.
#     if arrays[0].ndim == 1:
#         arrays[0] = np.atleast_2d(arrays[0])
#     if arrays[-1].ndim == 1:
#         arrays[-1] = np.atleast_2d(arrays[-1]).T
#     _assertRank2(*arrays)

#     # _multi_dot_three is much faster than _multi_dot_matrix_chain_order
#     if n == 3:
#         result = _multi_dot_three(arrays[0], arrays[1], arrays[2])
#     else:
#         order = _multi_dot_matrix_chain_order(arrays)
#         result = _multi_dot(arrays, order, 0, n - 1)

#     # return proper shape
#     if ndim_first == 1 and ndim_last == 1:
#         return result[0, 0]  # scalar
#     elif ndim_first == 1 or ndim_last == 1:
#         return result.ravel()  # 1-D
#     else:
#         return result


# def _multi_dot_three(A, B, C):
#     """
#     Find the best order for three arrays and do the multiplication.

#     For three arguments `_multi_dot_three` is approximately 15 times faster
#     than `_multi_dot_matrix_chain_order`

#     """
#     # cost1 = cost((AB)C)
#     cost1 = (A.shape[0] * A.shape[1] * B.shape[1] +  # (AB)
#              A.shape[0] * B.shape[1] * C.shape[1])   # (--)C
#     # cost2 = cost((AB)C)
#     cost2 = (B.shape[0] * B.shape[1] * C.shape[1] +  # (BC)
#              A.shape[0] * A.shape[1] * C.shape[1])   # A(--)

#     if cost1 < cost2:
#         return np.dot(np.dot(A, B), C)
#     else:
#         return np.dot(A, np.dot(B, C))


# def _multi_dot_matrix_chain_order(arrays, return_costs=False):
#     """
#     Return a np.array that encodes the optimal order of mutiplications.

#     The optimal order array is then used by `_multi_dot()` to do the
#     multiplication.

#     Also return the cost matrix if `return_costs` is `True`

#     The implementation CLOSELY follows Cormen, "Introduction to Algorithms",
#     Chapter 15.2, p. 370-378.  Note that Cormen uses 1-based indices.

#         cost[i, j] = min([
#             cost[prefix] + cost[suffix] + cost_mult(prefix, suffix)
#             for k in range(i, j)])

#     """
#     n = len(arrays)
#     # p stores the dimensions of the matrices
#     # Example for p: A_{10x100}, B_{100x5}, C_{5x50} --> p = [10, 100, 5, 50]
#     p = [a.shape[0] for a in arrays] + [arrays[-1].shape[1]]
#     # m is a matrix of costs of the subproblems
#     # m[i,j]: min number of scalar multiplications needed to compute A_{i..j}
#     m = np.zeros((n, n), dtype=np.double)
#     # s is the actual ordering
#     # s[i, j] is the value of k at which we split the product A_i..A_j
#     s = np.empty((n, n), dtype=np.intp)

#     for l in range(1, n):
#         for i in range(n - l):
#             j = i + l
#             m[i, j] = np.Inf
#             for k in range(i, j):
#                 q = m[i, k] + m[k + 1, j] + p[i] * p[k + 1] * p[j + 1]
#                 if q < m[i, j]:
#                     m[i, j] = q
#                     s[i, j] = k  # Note that Cormen uses 1-based index

#     return (s, m) if return_costs else s


# def _multi_dot(arrays, order, i, j):
#     """Actually do the multiplication with the given order."""
#     if i == j:
#         return arrays[i]
#     else:
#         return np.dot(_multi_dot(arrays, order, i, order[i, j]),
#                       _multi_dot(arrays, order, order[i, j] + 1, j))


def stack(tup_tup):
    """
    Stack arrays similar to matlab's "square bracket stacking".

            [A A; B B]

    Parameters
    ----------
    tup_tup : sequence of sequence of ndarrays
        1-D arrays are treated as row vectors.

    Returns
    -------
    stacked : ndarray
        The 2-D array formed by stacking the given arrays.

    See Also
    --------
    hstack : Stack arrays in sequence horizontally (column wise).
    vstack : Stack arrays in sequence vertically (row wise).
    dstack : Stack arrays in sequence depth wise (along third dimension).
    concatenate : Join a sequence of arrays together.
    vsplit : Split array into a list of multiple sub-arrays vertically.

    Examples
    --------
    >>> A = np.array([[1, 2, 3]])
    >>> B = np.array([[2, 3, 4]])
    >>> stack([A, B])
    array([[1, 2, 3, 2, 3, 4]])

    >>> A = np.array([[1, 2, 3]]).T
    >>> B = np.array([[2, 3, 4]]).T
    >>> stack([A, B])
    array([[1, 2],
           [2, 3],
           [3, 4]])

    >>> A = np.array([[1, 2, 3]])
    >>> B = np.array([[2, 3, 4]])
    >>> stack([[A, A], [B, B]])
    array([[1, 2, 3, 1, 2, 3],
           [2, 3, 4, 2, 3, 4]])

    >>> A = np.array([[1, 2, 3]])
    >>> B = np.array([[2, 3, 4]])
    >>> stack([[A], [B]])
    array([[1, 2, 3],
           [2, 3, 4]])

    >>> # 1-D vectors are treated as row arrays
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> stack([a, b])
    array([[1, 2, 3, 2, 3, 4]])

    >>> # 1-D vectors are treated as row arrays
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([2, 3, 4])
    >>> stack([[a, b], [a, b]])
    array([[1, 2, 3, 2, 3, 4],
           [1, 2, 3, 2, 3, 4]])

    >>> # a bit more complex
    >>> A = np.array([[1, 2, 3]])
    >>> B = np.array([[2, 3, 4, 5, 6, 7]])
    >>> c = np.array([8, 3, 4, 5, 6, 7])
    >>> d1 = np.array([9])
    >>> d2 = np.array([8, 7, 6, 5, 4])
    >>> stack([[A, A], [B], [c], [d1, d2]])
    array([[1, 2, 3, 1, 2, 3],
           [2, 3, 4, 5, 6, 7],
           [8, 3, 4, 5, 6, 7],
           [9, 8, 7, 6, 5, 4]])


    """
    if isinstance(tup_tup[0], (list, tuple)):
        result = np.vstack([np.hstack([a for a in row]) for row in tup_tup])
    else:
        result = np.hstack([a for a in tup_tup])
    return np.atleast_2d(result)
