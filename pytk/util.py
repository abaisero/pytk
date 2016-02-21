# import operator


class Hashable(object):
    @property
    def _key(self):
        raise Exception

    def __hash__(self):
        return hash(self._key)
        # return reduce(operator.xor, map(hash, self._key))

    def __eq__(self, other):
        isinstance(self, type(other)) and self._key == self._key
