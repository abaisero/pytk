from functools import wraps
import time

from .static import static


def once_every(n):
    """ run the wrapped function only every nth call after the first call """
    def outer(func):
        @wraps(func)
        @static(ncalls=0, ncalls_actual=0, ncalls_filtered=0)
        def inner(*args, **kwargs):
            if inner.ncalls % n == 0:
                func(*args, **kwargs)
                inner.ncalls_actual += 1
            else:
                inner.ncalls_filtered += 1
            inner.ncalls += 1
        return inner
    return outer


def once_every_timer(period):
    """ run the method only once within each period of time (in seconds) """
    def outer(func):
        @wraps(func)
        @static(last=None, ncalls=0, ncalls_actual=0, ncalls_filtered=0)
        def inner(*args, **kwargs):
            now = time.time()
            if inner.last is None or inner.last < now - period:
                inner.last = now
                func(*args, **kwargs)
                inner.ncalls_actual += 1
            else:
                inner.ncalls_filtered += 1
            inner.ncalls += 1
        return inner
    return outer
