from functools import partial, wraps
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


class memoizemethod(object):
    """cache the return value of a method
    
    This class is meant to be used as a decorator of methods. The return value
    from a given method invocation will be cached on the instance whose method
    was invoked. All arguments passed to a method decorated with memoize must
    be hashable.
    
    If a memoized method is invoked directly on its class the result will not
    be cached. Instead the method will be invoked like a static method:
    class Obj(object):
        @memoize
        def add_to(self, arg):
            return self + arg
    Obj.add_to(1) # not enough arguments
    Obj.add_to(1, 2) # returns 3, result is not cached
    """
    def __init__(self, func):
        self.func = func
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self.func
        return partial(self, obj)
    def __call__(self, *args, **kw):
        obj = args[0]
        try:
            cache = obj.__cache
        except AttributeError:
            cache = obj.__cache = {}
        key = (self.func, args[1:], frozenset(kw.items()))
        try:
            res = cache[key]
        except KeyError:
            res = cache[key] = self.func(*args, **kw)
        return res


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
    """ Re-compute property only conditionally to another property """
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


def static(**kwargs):
    """ static function variables """
    def decorate(f):
        for k in kwargs:
            setattr(f, k, kwargs[k])
        return f
    return decorate


def boolnot(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return not f(*args, **kwargs)
    return wrapper


class setter(property):
    """ Memoize a property such that it is computed only once """

    def __init__(self, fset, doc=None):
        super(setter, self).__init__(fget=None, fset=fset, doc=doc)


def monkeypatch(cls):
    def decorator(func):
        if cls is not None:
            try:
                fname = func.func_name
            except AttributeError:
                fname = func.__func__.func_name
            setattr(cls, fname, func)
        return func
    return decorator
