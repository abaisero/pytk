from abc import ABCMeta, abstractmethod

import inspect
import types


class Distribution(metaclass=ABCMeta):
    def __init__(self, *, cond=None):
        if cond is None:
            cond = ()
        xfactories = cond

        self.xfactories = xfactories
        self.nx = len(xfactories)

        self._dist = None
        self._pr = None
        self._sample = None
        self._E = None

    def def_dist(self, *, values=None, items=None):
        assert values is not None or items is not None
        if values is None: values = not items
        if items is None: items = not values
        assert bool(values) ^ bool(items)

        self._dist_values = values
        self._dist_map = self._tovalues if values else self._toitems
        def def_dist_(dist):
            assert len(inspect.signature(dist).parameters) == self.nx + 1
            self._dist = types.MethodType(dist, self)
            return dist
        return def_dist_

    def def_sample(self, *, values=None, items=None):
        assert values is not None or items is not None
        if values is None: values = not items
        if items is None: items = not values
        assert bool(values) ^ bool(items)

        self._sample_values = values
        self._sample_map = self._tovalues if values else self._toitems
        def def_sample_(sample):
            assert len(inspect.signature(sample).parameters) == self.nx + 1
            self._sample = types.MethodType(sample, self)
            return sample
        return def_sample_

    @abstractmethod
    def dist(self, *x):
        pass

    @abstractmethod
    def sample(self, *x):
        pass

    def _toitems(self, xs, factories):
        # return tuple(x if isinstance(x, f.Item) else f.item(value=x) for x, f in zip(xs, factories))
        return tuple(x if f.isitem(x) else f.item(value=x) for x, f in zip(xs, factories))

    def _tovalues(self, xs, factories):
        # return tuple(x.value if isinstance(x, f.Item) else x for x, f in zip(xs, factories))
        return tuple(x.value if f.isitem(x) else x for x, f in zip(xs, factories))

