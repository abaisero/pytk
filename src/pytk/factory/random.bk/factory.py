import logging
logger = logging.getLogger(__name__)

from .distribution import Distribution

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
    def asarray(self):
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

    # TODO other way of specifying?
    def def_pr(self, *, values=None, items=None):
        assert values is not None or items is not None
        if values is None: values = not items
        if items is None: items = not values
        assert bool(values) ^ bool(items)

        self._pr_values = values
        self._pr_map = self._tovalues if values else self._toitems
        def def_pr_(pr):
            assert len(inspect.signature(pr).parameters) == self.nxy + 1
            self._pr = types.MethodType(pr, self)
            return pr
        return def_pr_

    def dist(self, *x):
        logger.debug(f'dist() \t; x={x}')
        assert len(x) == self.nx

        # ensures inputs are items
        # TODO only use this when necessary!
        # WORK ON THIS TOMORROW!
        # x = self._toitems(x, self.xfactories)

        if self._dist is not None:
            x = self._dist_map(x, self.xfactories)
            dist = self._dist(*x)
        elif self._pr is not None:
            dist = ((*y, self.pr(*x, *y)) for y in itt.product(*self.yfactories))
            dist = (ypr for ypr in dist if ypr[-1] > 0.)
        else:
            raise NotImplementedError

        return dist

    def pr(self, *xy):
        logger.debug(f'pr() \t; xy={xy}')
        assert len(xy) == self.nxy

        # TODO only do conversions when necessary ensures inputs are items
        # xy = self._toitems(xy, self.xyfactories)

        if self._pr is not None:
            xy = self._pr_map(xy, self.xyfactories)
            pr = self._pr(*xy)
        elif self._dist is not None:
            x, y = xy[:self.nx], xy[self.nx:]
            pr = sum(yp[-1] for yp in self.dist(*x) if yp[:-1] == y)
        else:
            raise NotImplementedError

        return pr

    def sample(self, *x):
        logger.debug(f'sample() \t; x={x}')
        assert len(x) == self.nx

        # ensures inputs are items
        # TODO only do this when necessary
        # x = self._toitems(x, self.xfactories)

        if self._sample is not None:
            x = self._dist_map(x, self.xfactories)
            y = self._sample(*x)
        elif self._dist is not None or self._pr is not None:
            dist = list(self.dist(*x))
            ys = [yp[:-1] for yp in dist]
            ps = [yp[-1] for yp in dist]
            yi = rnd.choice(len(ys), p=ps)
            y = ys[yi]
        else:
            raise NotImplementedError

        # TODO figure out how to handle special case of single element
        # if isinstance(y, tuple) and len(y) == 1:
        #     y = y[0]

        # print(y)
        # y = self._dist_map(y, self.yfactories)
        # print(y)
        return y
