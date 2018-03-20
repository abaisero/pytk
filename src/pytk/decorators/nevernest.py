from functools import wraps


class NestingError(Exception):
    pass

def nevernest(n=1):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            if wrapper.__nnest == n:
                raise NestingError(f'Function has been nested {n} time(s)!')

            wrapper.__nnest += 1
            x = f(*args, **kwargs)
            wrapper.__nnest -= 1
            return x

        wrapper.__nnest = 0
        return wrapper
    return decorator
