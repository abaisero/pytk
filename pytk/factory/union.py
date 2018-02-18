from .factory import Factory
import numpy as np
import itertools as itt


# NOTE special case... not a standard factory!

class FactoryUnion(Factory):
    def __init__(self, **fmap):
        self.__fmap = fmap
        self.__fimap = {k: i for i, k in enumerate(fmap.keys())}

        dims = (0,) + tuple(f.nitems for f in fmap.values())
        self.__cumdims = np.cumsum(dims)
        self.nitems = self.__cumdims[-1]

    def __getattr__(self, name):
        try:
            return self.__fmap[name]
        except KeyError:
            raise AttributeError

    @property
    def values(self):
        for k, f in self.__fmap.items():
            for v in f.values:
                yield k, v

    @property
    def items(self):
        return itt.chain(*(f.items for f in self.__fmap.values()))

    def i(self, value):
        k, v = value
        fi = self.__fimap[k]
        i = self.__fmap[k].i(v)
        return self.__cumdims[fi] + i

    def value(self, i):
        for k, f in self.__fmap.items():
            if 0 <= i < f.nitems:
                return k, f.value(i)
            i -= f.nitems
        raise ValueError

    # simulating Item creation
    @staticmethod  # hack/trick;  self passed twice..
    def Item(self, i):
        for k, f in self.__fmap.items():
            if 0 <= i < f.nitems:
                return f.item(i)
            i -= f.nitems
        raise ValueError

    def isitem(self, item):
        """ overriding """
        return any(f.isitem(item) for f in self.__fmap.values())
