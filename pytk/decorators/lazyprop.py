class lazyprop(property):
    """ Memoize a property such that it is computed only once """

    def __init__(self, fget, doc=None):
        super(lazyprop, self).__init__(fget=fget, doc=doc)
        self.attr_name = '__lazy__{}'.format(fget.__name__)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError('unreadable attribute')
        if not hasattr(obj, self.attr_name):
            setattr(obj, self.attr_name, self.fget(obj))
        return getattr(obj, self.attr_name)

    def __set__(self, obj, value):
        if self.fget is None:
            raise AttributeError('unreadable attribute')
        setattr(obj, self.attr_name, value)

    def __delete__(self, obj):
        if self.fget is None:
            raise AttributeError('unreadable attribute')
        delattr(obj, self.attr_name)

    def getter(self, fget):
        return type(self)(fget, self.__doc__)

    def setter(self, fset):
        raise AttributeError('lazyprop is read-only and does not allow to set explicit setter.')

    def deleter(self, fdel):
        raise AttributeError('lazyprop is read-only and does not allow to set explicit deleter.')
