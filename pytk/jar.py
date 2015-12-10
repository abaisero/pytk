import cPickle as pickle
import dill

from .decorators import cache


@cache
def unpickle(fpath):
    """Unpickle path (but only if it's not done yet)."""
    with open(fpath, 'rb') as f:
        return pickle.load(f)


@cache
def undill(fpath):
    """Undill path (but only if it's not done yet)."""
    with open(fpath, 'rb') as f:
        return dill.load(f)


class Jar(object):

    def __init__(self):
        self.labels = []

    @property
    def label(self):
        return '.'.join(['Jar'] + self.labels + ['dill'])

    def clear(self):
        del self.labels[:]

    def seal(self, obj):
        with open(self.label, 'wb') as f:
            dill.dump(obj, f)

    def unseal(self):
        return undill(self.label)

    def __str__(self):
        return self.label
