import numpy as np

import pickle


def allUnique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)

from collections import OrderedDict


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
        self.tag2idx = OrderedDict()

    def subtags(self, t=''):
        return filter(lambda tag: tag.startswith(t), self.tag2idx.keys())

    def add(self, tag, n=1):
        self.tag2idx[tag] = np.arange(self.n, self.n + n)
        self.n += n

    def mask(self, *tags):
        if not tags:
            return np.ones(self.n, dtype=bool)
        mask = np.zeros(self.n, dtype=bool)
        tags = [t for tag in map(self.subtags, tags) for t in tag]
        for tag in tags:
            idx = self.tag2idx[tag]
            mask[idx] = True
        return mask

    def filter(self, a, *tags):
        mask = self.mask(*tags)
        return a[mask]

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)
