import collections


class defaultdict_noinsert(collections.defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError((key,))
        return self.default_factory()
