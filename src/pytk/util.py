import collections
# import operator

import numpy as np


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


# TODO destroy this madness...
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


def compose(*funcs):
    """ Composes input functions f(.), g(.), and h(.) into f(g(h(.))) """
    return lambda x: reduce(lambda v, f: f(v), reversed(funcs), x)




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
