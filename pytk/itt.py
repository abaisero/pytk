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
