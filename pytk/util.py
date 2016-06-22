# import operator


class Hashable(object):
    @property
    def _hashable_key(self):
        raise Exception

    def __hash__(self):
        return hash(self._hashable_key)
        # return reduce(operator.xor, map(hash, self._hashable_key))

    def __eq__(self, other):
        return isinstance(self, type(other)) and self._hashable_key == other._hashable_key


def tryconvert(x, t):
    """returns t(x) if no exception is raised. Otherwise x"""
    try:
        return t(x)
    except:
        pass
    return x
