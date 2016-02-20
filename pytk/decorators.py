from functools import wraps
from timeit import default_timer as timer


################################################################################
# Memoization
################################################################################

def memoize(f):
    """Decorator: cache the results of f for the same parameters.
    The decorated function is only called if the parameters differ from
    previous calls.
    Cache is really useful for recursive functions!

    Warning:
        Only use this with pure functions/functons without side effects.
    """
    cache = dict()

    @wraps(f)
    def wrapper(*args, **kwargs):
        key = tuple(args) + tuple(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = f(*args, **kwargs)
        return cache[key]
    return wrapper


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


class sentinel(object):
    """ Re-compute property conditionally to another property """
    class _sentinel_property(property):
        def __init__(self, fget, watch):
            super(sentinel._sentinel_property, self).__init__(fget=fget)
            self.watch = watch
            self.watch_cache = '__sentinel_watch_{}'.format(fget.__name__)
            self.watchman_cache = '__sentinel_watchman_{}'.format(fget.__name__)

        def __get__(self, obj, objtype=None):
            if obj is None:
                return self
            if self.fget is None:
                raise AttributeError('unreadable attribute')
            watch = getattr(obj, self.watch)
            if not hasattr(obj, self.watchman_cache) or getattr(obj, self.watch_cache) is not watch:
                setattr(obj, self.watch_cache, watch)
                setattr(obj, self.watchman_cache, self.fget(obj))
            return getattr(obj, self.watchman_cache)

        def __delete__(self, obj):
            delattr(obj, self.watchman_cache)

    def __init__(self, attr_name):
        self.attr_name = attr_name

    def __call__(self, f):
        return sentinel._sentinel_property(f, self.attr_name)

################################################################################
# Function execution control
################################################################################


def once_every_nth(n):
    """ run the wrapped function only every nth call after the first call """
    def inner(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if wrapper.n % wrapper.period == 0:
                f(*args, **kwargs)
            wrapper.n += 1
        wrapper.period = n
        wrapper.n = 0
        return wrapper
    return inner


def once_every_period(period):
    """ run the method only once within each period of time (in seconds) """
    def inner(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            now = timer()
            if wrapper.last is None or wrapper.last < now - wrapper.period:
                wrapper.last = now
                f(*args, **kwargs)
            wrapper.n += 1
        wrapper.period = period
        wrapper.last = None
        wrapper.n = 0
        return wrapper
    return inner


################################################################################
# Misc
################################################################################

# # TODO I have no idea why or how this came to be
# def static(**kwargs):
#     """ static function variables """
#     def decorate(f):
#         for k in kwargs:
#             setattr(f, k, kwargs[k])
#         return f
#     return decorate
