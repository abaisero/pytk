import logging
logger = logging.getLogger(__name__)

from .distribution import Distribution

from pytk.decorators import nevernest, NestingError

import itertools as itt
import inspect
import types

import numpy as np
import numpy.random as rnd

# TODO let the def_ methods return values directly instead of having to return items!
# TODO let the inputs also be values?!  probably not...?


# TODO maybe separate thing for single dimension y?
class FactoryDistribution(Distribution):
    logger = logging.getLogger(f'{__name__}.FactoryDistribution')

    def __init__(self, *yfactories, cond=None):
        super().__init__(cond=cond)

        self.yfactories = yfactories
        self.xyfactories = self.xfactories + self.yfactories

        self.ny = len(self.yfactories)
        self.nxy = len(self.xyfactories)

        # TODO use array as cache for other methods
        self.__array = None
        # TODO also allow the user to specify distribution as array...
        # TODO what about terminal states?

    @property
    def array(self):
        if self.__array is None:
            shape = tuple(f.nitems for f in self.xyfactories)
            self.__array = np.zeros(shape)
            for x in itt.product(*self.xfactories):
                xi = tuple(item.i for item in x)
                for yp in self.dist(*x):
                    y, p = yp[:-1], yp[-1]
                    yi = tuple(item.i for item in y)
                    self.__array[xi+yi] += p

        return self.__array

    @array.setter
    def array(self, value):
        assert value.shape == tuple(f.nitems for f in self.xyfactories)
        self.__array = value

    # TODO in all methods, use array if exists...
    # TODO for dist and pr methods;  give a* interface

    # TODO use array if exists...

    def adist(self, *x):
        logger.debug(f'adist() \t; x={x}')
        assert len(x) == self.nx

        shape = tuple(f.nitems for f in self.yfactories)
        adist = np.zeros(shape)
        for yp in self.dist(*x):
            y, p = yp[:-1], yp[-1]
            yi = tuple(item.i for item in y)
            adist[yi] += p

        return adist

    # TODO probably better wait..
    @nevernest(n=1)
    def dist(self, *x):
        logger.debug(f'dist() \t; x={x}')
        assert len(x) == self.nx

        if self.__array is not None:
            for y in itt.product(*self.yfactories):
                idx = tuple(item.i for item in (x+y))
                yield (*y, self.array[idx])
            return

        try:
            dist = ((*y, self.pr(*x, *y)) for y in itt.product(*self.yfactories))
            dist = (ypr for ypr in dist if ypr[-1] > 0.)
        except NestingError as e:
            raise NotImplementedError('Method self.dist() of this distribution was neither supplied nor can it be computed automagically.') from e

        return dist

    # TODO probably better wait..
    @nevernest(n=1)
    def pr(self, *xy):
        logger.debug(f'pr() \t; xy={xy}')
        assert len(xy) == self.nxy

        try:
            x, y = xy[:self.nx], xy[self.nx:]
            pr = sum(yp[-1] for yp in self.dist(*x) if yp[:-1] == y)
        except NestingError as e:
            raise NotImplementedError('Method self.pr() of this distribution was neither supplied nor can it be computed automagically.') from e

        return pr

    # TODO probably better wait..
    @nevernest(n=1)
    def sample(self, *x):
        logger.debug(f'sample() \t; x={x}')
        assert len(x) == self.nx

        try:
            dist = list(self.dist(*x))
            ys = [yp[:-1] for yp in dist]
            ps = [yp[-1] for yp in dist]
            yi = rnd.multinomial(1, ps).argmax()
            # yi = rnd.choice(len(ys), p=ps)
            y = ys[yi]
        except NestingError as e:
            raise NotImplementedError('Method self.sample() of this distribution was neither supplied nor can it be computed automagically.') from e

        return y
