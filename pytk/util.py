import collections
# import operator

import numpy as np
import numpy.random as rnd


# NOTE deprecated.  Use Keyable instead
class Hashable(object):
    @property
    def _hashable_key(self):
        raise Exception

    def __hash__(self):
        return hash(self._hashable_key)
        # return reduce(operator.xor, map(hash, self._hashable_key))

    def __eq__(self, other):
        return isinstance(self, type(other)) and self._hashable_key == other._hashable_key


class Keyable(object):
    """ Better version of Hashable """
    # # TODO either directly give values, or give attr names
    def setkey(self, values, names=False):
        self.__values = values
        self.__names = names

    @property
    def __key(self):
        return (tuple(getattr(self, name) for name in self.__values)
                if self.__names
                else self.__values)

    def __hash__(self):
        return hash(self.__key)

    def __eq__(self, other):
        return (self.__key == other.__key
                if isinstance(other, type(self))
                else NotImplemented)

    def __ne__(self, other):
        result = self.__eq__(other)
        return (result
                if result is NotImplemented
                else not result)


def trymap(f, x):
    """returns f(x) if no exception is raised. Otherwise x"""
    try:
        return f(x)
    except:
        return x


def compose(*funcs):
    """ Composes input functions f(.), g(.), and h(.) into f(g(h(.))) """
    return lambda x: reduce(lambda v, f: f(v), reversed(funcs), x)


def argmax(f, xs, all_=False, rnd_=False):
    if all_ and rnd_:
        raise ValueError('Arguments `all_` and `rnd_` can not both be true.')

    fs = map(f, xs)

    if not all_ and not rnd_:
        xi = np.argmax(fs)
        return xs[xi]

    fmax = None
    for x, f in zip(xs, fs):
        if fmax is None or f > fmax:
            xmaxs, fmax = [], f
        if fmax == f:
            xmaxs.append(x)

    if all_:
        return xmaxs
    # if rnd_:  # has to be true
    xi = rnd.choice(len(xmaxs))
    return xmaxs[xi]


class true_every(object):
    def __init__(self, n):
        self.i = 0
        self.n = n

    def __nonzero__(self):
        b = not self.i % self.n
        self.i += 1
        return b

    @property
    def true(self):
        return bool(self)

    @property
    def false(self):
        return not self.true
