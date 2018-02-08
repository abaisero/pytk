from .factory import Factory


class FactorySubset(Factory):
    class Item(Factory.Item):
        def change(self, include, exclude):
            include = set(include)
            exclude = set(exclude)
            self.value = (self.value - exclude) | (include - exclude)

        def include(self, values):
            self.value = self.value.union(values)

        def exclude(self, values):
            self.value = self.value.difference(values)

    def __init__(self, vlist):
        self.vlist = vlist
        self.nbits = len(vlist)
        self.nitems = 1 << self.nbits

    def i(self, value):
        return sum(1 << self.vlist.index(v) for v in value)

    def value(self, i):
        bits = (k for k in range(self.nbits) if i >> k & 1)
        return set(self.vlist[bit] for bit in bits)

    @property
    def values(self):
        for i in range(self.nitems):
            yield self.value(i)
