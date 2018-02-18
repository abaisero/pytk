from functools import wraps


def once_every_nth(n):
    """ run the wrapped function only every nth call after the first call """
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if wrapper.n % wrapper.period == 0:
                f(*args, **kwargs)
            wrapper.n += 1
        wrapper.period = n
        wrapper.n = 0
        return wrapper
    return decorator


def once_every_period(period):
    """ run the method only once within each period of time (in seconds) """
    def decorator(f):
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
    return decorator
