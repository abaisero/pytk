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
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        if self.fget is None:
            raise AttributeError('unreadable attribute')
        pname = '__lazy__{}'.format(self.fget.__name__)
        if getattr(obj, pname, None) is None:
            setattr(obj, pname, self.fget(obj))
        return getattr(obj, pname)

def static(**kwargs):
    """ static function variables """
    def decorate(f):
        for k in kwargs:
            setattr(f, k, kwargs[k])
        return f
    return decorate

class Every(object):
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

