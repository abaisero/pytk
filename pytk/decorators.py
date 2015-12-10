from functools import wraps


def cache(f):
    """Decorator: cache the results of f for the same parameters.
    The decorated function is only called if the parameters differ from
    previous calls.
    Cache is really useful for recursive functions!

    Warning:
        Only use this with pure functions/functons without side effects.
    """
    saved = {}

    @wraps(f)
    def wrapper(*args):
        if args not in saved:
            saved[args] = f(*args)
        return saved[args]
    return wrapper


class lazyprop(property):

    def __init__(self, fget, doc=None):
        super(lazyprop, self).__init__(fget=fget, doc=doc)
        self.pname = '__lazy__{}'.format(fget.__name__)

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError('unreadable attribute')
        if not hasattr(obj, self.pname):
            setattr(obj, self.pname, self.fget(obj))
        return getattr(obj, self.pname)

    def __set__(self, obj, value):
        if self.fget is None:
            raise AttributeError('unreadable attribute')
        setattr(obj, self.pname, value)

    def __del__(self, obj):
        if self.fget is None:
            raise AttributeError('unreadable attribute')
        delattr(obj, self.pname)

    def getter(self, fget):
        return type(self)(fget, self.__doc__)

    def setter(self, fset):
        raise AttributeError('Lazyprop does not allow to set explicit setter')

    def deleter(self, fdel):
        raise AttributeError('Lazyprop does not allow to set explicit deleter')


def static(**kwargs):
    """ static function variables """
    def decorate(f):
        for k in kwargs:
            setattr(f, k, kwargs[k])
        return f
    return decorate


class every_nth(object):
    """ run the method only every nth call (first one included) """

    def __init__(self, period):
        self.period = period
        self.n = 0

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if self.n % self.period == 0:
                f(*args, **kwargs)
            self.n += 1
        return wrapper

from timeit import default_timer as timer


class every(object):

    def __init__(self, s, m=0, h=0):
        self.s = s + 60 * m + 3600 * h
        self.last = None

    def __call__(self, f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            now = timer()
            if self.last is None or self.last + self.s <= now:
                self.last = now
                f(*args, **kwargs)
        return wrapper


class _sentinel_property(property):

    def __init__(self, fget, watch):
        super(_sentinel_property, self).__init__(fget=fget)
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


class sentinel(object):

    def __init__(self, pname):
        self.pname = pname

    def __call__(self, f):
        return _sentinel_property(f, self.pname)
