from .model import Model

import numpy as np
import numpy.random as rnd
from scipy.misc import logsumexp

import string


class Softmax(Model):
    def __init__(self, *yfactories, cond=None):
        super().__init__(*yfactories, cond=cond)
        self.xshape = tuple(f.nitems for f in self.xfactories)
        self.yshape = tuple(f.nitems for f in self.yfactories)
        self.shape = self.xshape + self.yshape
        # NOTE np.prod would return float 1.0 if xshape is empty
        self.xsize = np.prod(self.xshape, dtype=np.int64)
        self.ysize = np.prod(self.yshape)
        self.size = self.xsize * self.ysize
        self.xaxis = tuple(range(self.nx))
        self.yaxis = tuple(range(self.nx, self.nxy))

        #  subscripts for np.einsum
        self.xss = string.ascii_lowercase[:self.nx]
        self.yss = string.ascii_lowercase[self.nx:self.nxy]
        self.xyss = string.ascii_lowercase[:self.nxy]

        # precomputed once
        self.__phi = np.eye(self.size).reshape(2 * self.shape)

        self.reset()

    def reset(self):
        self.params = np.zeros(self.shape)
        # self.params = rnd.normal(size=self.shape)

    @staticmethod
    def index(item, *, keepdims=False):
        if item is None:
            return slice(None)
        if item is Ellipsis:
            return slice(None)
        if isinstance(item, slice):
            return item
        if keepdims:  # NOTE Not currently being used
            return slice(item.i, item.i+1)  # keeps dimensions when indexing
        # Assumes item is an Item
        return item.i

    def indices(self, *items, keepdims=False):
        items += (None,) * (self.nxy - len(items))
        return tuple(self.index(item, keepdims=keepdims) for item in items)

    def xyindices(self, *items):
        idx = self.indices(*items)
        return idx[:self.nx], idx[self.nx:]

    def xyitems(self, *items):
        return items[:self.nx], items[self.nx:]

    def prefs(self, *items):
        idx = self.indices(*items)
        return self.params[idx]

    def logprobs(self, *items, normalized=False):
        idx = self.indices(*items)

        prefs = self.params
        logprobs = prefs
        if normalized:
            logprobs -= logsumexp(prefs, axis=self.yaxis, keepdims=True)

        return logprobs[idx]

    def probs(self, *items):
        logprobs = self.logprobs(*items)
        probs = np.exp(logprobs - logprobs.max())
        # TODO future bug! only normalize the y axes which were not given!!
        return probs / probs.sum()

    # all the next methods assume that all items are given!!
    # generalize! this will simplify logprob!


    def phi(self, *items):
        idx = self.indices(*items)
        return self.__phi[idx]

    def dprefs(self, *items):
        return self.phi(*items)

    # def dlogprobs(self, *items):
    #     xidx, _ = self.xyindices(*items)
    #     idx = self.indices(*items)

    #     dprefs = self.dprefs()
    #     probs = self.probs()

    #     # subscripts = f'{self.xyss},{self.xyss}...->{self.xss}...'
    #     # Edprefs = np.einsum(subscripts, probs, dprefs)

    #     # dlogprobs = dprefs[idx] - Edprefs[xidx]
    #     # return dlogprobs

    #     subscripts = f'{self.yss},{self.yss}...->...'
    #     Edprefs = np.einsum(subscripts, probs[xidx], dprefs[xidx])

    #     dlogprobs = dprefs[idx] - Edprefs
    #     return dlogprobs

    def dlogprobs(self, *items):
        xitems, _ = self.xyitems(*items)
        xidx, yidx = self.xyindices(*items)
        idx = self.indices(*items)

        dprefs = self.dprefs(*xitems)
        probs = self.probs(*xitems)
        dlogprobs = dprefs[yidx] - np.tensordot(probs, dprefs, axes=self.ny)
        return dlogprobs

    # TODO I don't know if I need this... if I do, it's not hard to implement
    # def dprobs(self, *items):
    #     probs = self.probs(*items))
    #     dlogprobs = self.dlogprobs(*items)

    # def ddprefs(self, *items):
    #     idx = self.indices(*items)

    #     ddprefs = np.zeros(3 * self.shape)
    #     return ddprefs[idx]


    # def ddlogprobs(self, *items):
    #     dprefs = self.dprefs()
    #     probs = self.probs()

    #     dprefs2 =
    #     Edprefs = np.tensordot(probs, dprefs, axes=(self.yaxis, self.yaxis))

    #     # collapsing xidx indices
    #     subscripts = f'{self.xss}{self.xss}...->{self.xss}...'
    #     Edprefs = np.einsum(subscripts, Edprefs)

    #     Edprefs_2 =
    #     E_dprefs2 =

    #     ddlogprobs = Edprefs * Edprefs - Edprefs_2
    #     return ddlogprobs[idx]

    def dist(self, *xitems):
        assert len(xitems) == self.nx

        probs = self.probs(*xitems)
        for yi in range(self.ysize):
            yidx = np.unravel_index(yi, self.yshape)
            yitems = tuple(f.item(i) for f, i in zip(self.yfactories, yidx))
            yield yitems + (probs[yidx],)

    def pr(self, *items):
        assert len(items) == self.nxy

        return self.probs(*items)

    def sample(self, *xitems):
        assert len(xitems) == self.nx

        # TODO kinda like a JointFactory but without names;  just indices?

        probs = self.probs(*xitems).ravel()
        # yi = rnd.choice(self.ysize, p=probs)
        yi = rnd.multinomial(1, probs).argmax()
        yidx = np.unravel_index(yi, self.yshape)
        yitems = tuple(f.item(i) for f, i in zip(self.yfactories, yidx))

        if len(yitems) == 1:
            return yitems[0]
        return yitems
