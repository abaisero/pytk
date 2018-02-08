from .factory import Factory

from collections import namedtuple
import numpy as np


class FactoryJoint(Factory):
    class Item(Factory.Item):
        @property
        def i(self):
            indices = tuple(s.i for s in self.imap.values())
            return np.ravel_multi_index(indices, self.factory.dims)

        @i.setter
        def i(self, ii):
            try:
                # TODO use factory method, don't duplicate code..
                indices = np.unravel_index(ii, self.factory.dims)
                for i, item in zip(indices, self.imap.values()):
                    item.i = i
            except AttributeError:
                self.imap = self.factory.imap(ii)

        def __getattr__(self, name):
            try:
                return super().__dict__['imap'][name]
            except KeyError:
                raise AttributeError

    def __init__(self, **fmap):
        self.fmap = fmap
        self.dims = tuple(sf.nitems for sf in fmap.values())
        self.nitems = np.prod(self.dims)

        self.vtype = namedtuple('Value', fmap.keys())

    def __getattr__(self, name):
        try:
            return self.fmap[name]
        except KeyError:
            raise AttributeError

    @property
    def values(self):
        return (self.value(i) for i in range(self.nitems))

    def i(self, value):
        # TODO changed this such that it accepts a dictionary, not a list
        # TODO maybe change it again such that it accepts keywords directly?
        indices = tuple(f.i(value[k]) for k, f in self.fmap.items())
        return np.ravel_multi_index(indices, self.dims)

    def value(self, i):
        indices = np.unravel_index(i, self.dims)
        vmap = {
            fkey: f.value(k) for (fkey, f), k in zip(self.fmap.items(), indices)
        }
        return self.vtype(**vmap)

    def imap(self, i):
        indices = np.unravel_index(i, self.dims)
        return {
           fkey: f.item(k) for (fkey, f), k in zip(self.fmap.items(), indices)
        }
