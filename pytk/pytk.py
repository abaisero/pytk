import numpy as np

import cPickle as pickle


def allUnique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)


class PostIt(object):
    """
    Book-keeping utility for sequential data.

    Allows to apply hierarchical tags to a sequential structure
    (e.g. numpy.ndarray) and to extract corresponding parts.

    Examples
    --------
    >>> pi = PostIt()
    >>> pi.add('a.x.1', 2)
    >>> pi.add('a.x.2', 3)
    >>> pi.add('b.y.1', 4)
    >>> pi.add('a.y.2', 5)

    >>> arr = np.arange(14)
    >>> pi.filter(arr)
    array([ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    >>> pi.filter(arr, 'a')
    array([ 0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
    >>> pi.filter(arr, 'a.x')
    array([ 0, 1, 2, 3, 4])
    >>> pi.filter(arr, 'a.x.1')
    array([ 0, 1])
    >>> pi.filter(arr, 'a.x.2')
    array([ 2, 3, 4])

    """
    def __init__(self):
        self.n = 0
        self.__idx = {}

    def add(self, tag, n=1):
        self.__idx[tag] = np.arange(self.n, self.n + n)
        self.n += n

    def mask(self, tag=None):
        if tag is None:
            return np.ones(self.n, dtype=bool)
        if tag in self.__idx:
            mask = np.zeros(self.n, dtype=bool)
            mask[self.__idx[tag]] = True
            return mask
        return np.logical_or.reduce([self.mask(k) for k in self.__idx.keys() if k.startswith(tag)])

    def filter(self, a, tag=None):
        print a.shape, self.mask(tag).shape
        return a[self.mask(tag)]

    @classmethod
    def load(cls, fname):
        with open(fname, 'r') as f:
            return pickle.load(f)

    def save(self, fname):
        with open(fname, 'w') as f:
            pickle.dump(self, f)
