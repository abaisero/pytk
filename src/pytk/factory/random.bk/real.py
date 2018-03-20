import logging
logger = logging.getLogger(__name__)

from .distribution import Distribution

import itertools as itt
import inspect
import types

import numpy as np
import numpy.random as rnd


class RealDistribution(Distribution):
    logger = logging.getLogger(f'{__name__}.RealDistribution')

    def __init__(self, *, cond=None):
        super().__init__(cond=cond)
        self.__array = None

    @property
    def asarray(self):
        if self.__array is None:
            shape = tuple(f.nitems for f in self.xfactories)
            self.__array = np.zeros(shape)
            for x in itt.product(*self.xfactories):
                xi = tuple(item.i for item in x)
                self.__array[xi] = self.E(*x)

        return self.__array

    def def_E(self, *, values=None, items=None):
        assert values is not None or items is not None
        if values is None: values = not items
        if items is None: items = not values
        assert bool(values) ^ bool(items)

        self._E_values = values
        self._E_map = self._tovalues if values else self._toitems
        def def_E_(E):
            assert len(inspect.signature(E).parameters) == self.nx + 1
            self._E = types.MethodType(E, self)
            return E
        return def_E_

    def dist(self, *x):
        logger.debug(f'dist() \t; x={x}')
        assert len(x) == self.nx

        # TODO only use this when necessary
        x = self._toitems(x, self.xfactories)

        if self._dist is not None:
            x = self._dist_map(x, self.xfactories)
            dist = self._dist(*x)
        else:
            raise NotImplementedError

        return dist

    def sample(self, *x):
        logger.debug(f'sample() \t; x={x}')
        assert len(x) == self.nx

        # TODO only use this when necessary
        x = self._toitems(x, self.xfactories)

        if self._sample is not None:
            x = self._sample_map(x, self.xfactories)
            r = self._sample(*x)
        elif self._dist is not None:
            dist = list(self.dist(*x))
            rs = [r for r, _ in dist]
            ps = [p for _, p in dist]
            r = rnd.choice(rs, p=ps)
        else:
            raise NotImplementedError

        return r

    def E(self, *x):
        logger.debug(f'E() \t; x={x}')
        assert len(x) == self.nx

        # TODO only use this when necessary
        x = self._toitems(x, self.xfactories)

        if self._E is not None:
            x = self._E_map(x, self.xfactories)
            E = self._E(*x)
        elif self._dist is not None:
            dist = self.dist(*x)
            E = sum(r * p for r, p in dist)
        else:
            raise NotImplementedError

        return E
