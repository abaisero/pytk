import collections


class WriteOnceKeyError(Exception):
    """ raise when key already exists in WriteOnceDict """


class WriteOnceDict(collections.MutableMapping):
    """ Dictionary-like structure which doesn't allow to overwrite entries. """

    def __init__(self, *args, **kwargs):
        self.store = dict()
        self.update(dict(*args, **kwargs))

    def __getitem__(self, k):
        return self.store[k]

    def __setitem__(self, k, v):
        try:
            self.store[k]
        except KeyError:
            self.store[k] = v
        else:
            raise WriteOnceKeyError(k)

    def __delitem__(self, k):
        del self.store[k]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)
