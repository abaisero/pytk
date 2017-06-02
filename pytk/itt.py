import itertools as itt

# both use virtually the same time for 10-1000000 (but ugly one SLIGHTLY faster)
# python -m timeit "it=iter(range(10000)); sentinel=object(); next(it, sentinel) is sentinel"
# python -m timeit "it=range(10000); any(True for _ in it)"
# def empty(iterable):
#     sentinel = object()
#     return next(iter(iterable), sentinel) is sentinel
def empty(iterable):
    return all(False for _ in iterable)

def izipcount(*iterables):
    return itt.izip(itt.count(), *iterables)

def nth(n, iterable):
    if n < 0:
        return IndexError('list index out of range')
    for i, value in enumerate(iterable):
        if i == n:
            return value
    return IndexError('list index out of range')

def grid(shape):
    return itt.product(*(xrange(k) for k in shape))

