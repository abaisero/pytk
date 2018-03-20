from scipy.misc import logsumexp


def normal(a, *, log=False):
    if log:
        return a - logsumexp(a)
    return a / a.sum()


def conditional(a, axis, *, log=False):
    try:
        axis = tuple(axis)
    except TypeError:
        axis = (axis,)
    finally:
        axis = tuple(i for i in range(a.ndim) if i not in axis)
    if log: return a - logsumexp(a, axis, keepdims=True)
    return a / a.sum(axis, keepdims=True)


def marginal(a, axis, *, log=False):
    if log:
        return logsumexp(a, axis)
    return a.sum(axis=axis)


if __name__ == '__main__':
    import numpy as np

    a = np.arange(24).reshape((2, 3, 4))

    print(a)

    log = False

    b = normal(a, log=log)
    # b = np.exp(b)
    print('normal')
    print(b)
    print(b.shape)
    print(b.sum())
    print()

#     a = b

    b = conditional(a, 0, log=log)
    # b = np.exp(b)
    print('conditional')
    print(b)
    print(b.shape)
    print(b.sum(1))
    print()

    b = marginal(a, 0, log=log)
    # b = np.exp(b)
    print('marginal')
    print(b)
    print(b.shape)
    print(b.sum())
