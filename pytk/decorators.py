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

# class lazy(property):
#     def __init__(self, fget=None, fset=None, fdel=None, doc=None):
#         super(lazy, self).__init__(fget, fset, fdel, doc)
#         self._mangled_property_name = '_lazy_{}_beenthere_donethat'.format(fget.__name__)

#     def __get__(self, obj, objtype=None):
#         if obj is None:
#             return self
#         if self.fget is None:
#             raise AttributeError("unreadable attribute")
#         if getattr(obj, self._mangled_property_name, None) is None:
#             setattr(obj, self._mangled_property_name, self.fget(obj))
#             print 'precomputing'
#         print 'computed'
#         return getattr(obj, self._mangled_property_name)

def lazyprop(fn):
    attr_name = '__lazyproperty__' + fn.__name__

    @property
    def lazy_wrapped(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)
    return lazy_wrapped

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

