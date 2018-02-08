from .factory import Factory


class FactoryFilter(Factory):
    class Item(Factory.Item):
        @property
        def i(self):
            fi = self.fitem.i
            return self.factory.fitoi(fi)

        @i.setter
        def i(self, ii):
            try:
                self.fitem.fi = self.factory.itofi(ii)
            except AttributeError:
                self.fitem = self.factory.fitem(ii)

        def __getattr__(self, name):
            try:
                return getattr(super(Item, self).__dict__['fitem'], name)
            except KeyError:
                raise AttributeError

    # @staticmethod
    # def ItemClass(factory):
    #     class Item(factory.Item):
    #         @property
    #         def i(self):
    #             # TODO convert from filtered index to filterless
    #             raise NotImplementedError

    #         @i.setter
    #         def i(self, ii):
    #             # TODO which type of assignment do I want to make...?
    #             raise NotImplementedError

    def __init__(self, factory, vfilter):
        # self.Item = self.ItemClass(factory)
        # TODO FUCK FUCK FUCK FUCK FUCK FUCK WHAT IS THE PROBLEM???

        # problem is that the item in this class is just the item of the parent
        # item class..

        self.factory = factory
        self.vfilter = vfilter

        vfilters = tuple(vfilter(v) for v in factory.values)
        ilist = (i for i, vf in enumerate(vfilters) if vf)

        self.ltarray = np.cumsum([not vf for vf in vfilters])
        self.iarray = np.array(list(ilist))
        self.nitems = factory.nitems - self.ltarray[-1]

    # TODO I think this is wrong....
    # TODO this is getting the imap from the factory..... fuck?
    def __getattr__(self, name):
        return getattr(self.factory, name)

    def i(self, value):
        if not self.vfilter(value):
            raise ValueError('Value {} not valid.'.format(value))
        i = self.factory.i(value)
        return i - self.ltarray[i]

    def value(self, i):
        try:
            fi = self.iarray[i]
        except IndexError as e:
            re = ValueError('Index {} not valid;  Must be < {}.'.format(i, self.nitems))
            raise re.with_traceback(e.__traceback__)
        else:
            return self.factory.value(fi)

    @property
    def values(self):
        for v in self.factory.values:
            if self.vfilter(v):
                yield v

    def itofi(self, i):
        return self.iarray[i]

    def fitoi(self, fi):
        return fi - self.ltarray[fi]

    def fitem(self, i):
        fi = self.itofi(i)
        return self.factory.item(fi)

    # def item(self, i=0):
    #     i -= self.ltarray[i]
    #     return self.factory.item(i)  # this will make it wrong?
        # return self.factory.Item(self, i)
