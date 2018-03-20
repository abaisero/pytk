import pytk.pack as pack

import numpy as np

import pickle


def allUnique(x):
    seen = set()
    return not any(i in seen or seen.add(i) for i in x)

from collections import OrderedDict


class PostIt(pack.Serializable):
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

    @property
    def _key(self):
        return (self.n, tuple(self.tag2idx.items()))

    def _encode(self):
        return dict(n=self.n, tag2idx_items=tuple(self.tag2idx.items()))

    @classmethod
    def _decode(cls, data):
        keys = ['n', 'tag2idx_items']
        n, tag2idx_items = (data[key] for key in keys)
        tag2idx = OrderedDict(tag2idx_items)
        obj = cls()
        obj.n = n
        obj.tag2idx = tag2idx
        return obj

    def __init__(self):
        self.n = 0
        self.tag2idx = OrderedDict()

    @property
    def tags(self):
        return ' '.join(('{}/{}'.format(k, v.size) for k, v in self.tag2idx.iteritems()))

    def subtags(self, tag=None):
        """ tag=None is special case, returns all the subtags instead of none """
        if tag is None:
            return self.tag2idx.keys()
        ltag = len(tag)
        def is_subtag(t):
            try:
                return t.startswith(tag) and t[ltag] == '.'
            except IndexError:
                return True
        return filter(is_subtag, self.tag2idx.keys())

    def add(self, tag, n=1):
        self.tag2idx[tag] = np.arange(self.n, self.n + n)
        self.n += n

    def imask(self, *tags):
        m = self.mask(*tags)
        return np.arange(len(m))[m]

    def mask(self, *tags):
        mask = np.zeros(self.n, dtype=bool)
        subtags = map(self.subtags, tags)
        tags = sum(subtags, [])
        for tag in tags:
            idx = self.tag2idx[tag]
            mask[idx] = True
        return mask

    def filter(self, a, *tags, **kwargs):
        full = kwargs.get('full', False)
        mask = self.mask(*tags)
        if full:
            value = np.copy(a)
            value[~mask] = 0
            return value
        return a[mask]

    @classmethod
    def load(cls, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def save(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(self, f)

    def __str__(self):
        return '(Postit {})'.format(self.tags)
