from functools import wraps


def astype(_type):
    def outer(func):
        @wraps(func)
        def inner(*args, **kwargs):
            return _type(func(*args, **kwargs))
        return inner
    return outer


def aslist(func):
    return astype(list)(func)

def astuple(func):
    return astype(tuple)(func)

def asdict(func):
    return astype(dict)(func)
