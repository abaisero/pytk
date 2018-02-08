from .factory import Factory

import numpy as np


class FactoryUnion(Factory):
    def __init__(self, factories):
        self.factories = factories
        self.dims = tuple(sf.nitems for f in factories)
        self.cumdims = np.cumsum(self.dims)
        self.nitems = self.cumdims[-1]
        # TODO check that all values are distinct

    def i(self, value):
        # TODO this only finds first value...
        si = None
        for sfi, sf in enumerate(self.factories):
            try:
                si = sf.i(value)
            except ValueError:
                pass
            else:
                break
        if si is None:
            raise ValueError
        i = si
        if sfi > 0:
            i += self.cumdims[sfi - 1]
        return i

    def value(self, i):
        sfi = (self.cumdims <= i).sum()
        si = i
        if sfi > 0:
            si -= self.cumdims[sfi - 1]
        return self.factories[sfi].value(si)

