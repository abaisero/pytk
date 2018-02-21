import logging
logger = logging.getLogger(__name__)

from .distribution import Distribution

from pytk.decorators import nevernest, NestingError

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

    @nevernest(n=1)
    def dist(self, *x):
        logger.debug(f'dist() \t; x={x}')
        assert len(x) == self.nx

        raise NotImplementedError('Method self.dist() of this distribution was neither supplied nor can it be computed automagically.')

    @nevernest(n=1)
    def sample(self, *x):
        logger.debug(f'sample() \t; x={x}')
        assert len(x) == self.nx

        try:
            dist = list(self.dist(*x))
            rs = [r for r, _ in dist]
            ps = [p for _, p in dist]
            ri = rnd.multinomial(1, ps).argmax()
            r = rs[ri]
            # r = rnd.choice(rs, p=ps)
        except NestingError as e:
            raise NotImplementedError('Method self.sample() of this distribution was neither supplied nor can it be computed automagically.') from e

        return r

    # TODO probably better wait..
    @nevernest(n=1)
    def E(self, *x):
        logger.debug(f'E() \t; x={x}')
        assert len(x) == self.nx

        try:
            dist = self.dist(*x)
            E = sum(r * p for r, p in dist)
        except NestingError as e:
            raise NotImplementedError('Method self.E() of this distribution was neither supplied nor can it be computed automagically.') from e

        return E
