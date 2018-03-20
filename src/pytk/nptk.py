import numpy as np

from contextlib import contextmanager


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


@contextmanager
def printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    yield
    np.set_printoptions(**original)
