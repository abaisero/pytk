import collections
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


def trymap(f, x):
    """returns f(x) if no exception is raised. Otherwise x"""
    try: return f(x)
    except: return x


def compose(*funcs):
    """ Composes input functions f(.), g(.), and h(.) into f(g(h(.))) """
    return lambda x: reduce(lambda v, f: f(v), reversed(funcs), x)
