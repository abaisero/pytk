value_None = object()


class FactoryException(Exception):
    pass


class Factory:
    class Item:
        def __init__(self, factory, i):
            self.factory = factory
            self.i = i

        @property
        def value(self):
            return self.factory.value(self.i)

        @value.setter
        def value(self, value):
            self.i = self.factory.i(value)

        def copy(self):
            return self.factory.item(self.i)

        def __eq__(self, other):
            try:
                return self.factory is other.factory and self.i == other.i
            except AttributeError:
                return self.value == other

        def __ne__(self, other):
            return not self == other

        def __hash__(self):
            return hash(self.factory) ^ hash(self.i)

        def __int__(self):
            return self.i

        def __str__(self):
            return self.factory.istr(self)

        def __repr__(self):
            return f'Item({self.factory.istr(self)})'

    @staticmethod
    def istr(item):
        return str(item.value)

    def i(self, value):
        raise NotImplementedError

    def item(self, i=None, value=value_None):
        if not self.check_ivalue(i, value):
            raise FactoryException('factory.item():  index and values do not match')

        if i is None:
            i = 0 if value is value_None else self.i(value)
        return self.Item(self, i)  # this might be annoying for union...

    def check_ivalue(self, i, value):
        return i is None or value is value_None or self.value(i) == value

    def isitem(self, item):
        try:
            return item.factory is self and 0 <= item.i < self.nitems
        except AttributeError:
            return False

    @property
    def items(self):
        return map(self.item, range(self.nitems))

    def __iter__(self):
        return self.items

    def __len__(self):
        return self.nitems
