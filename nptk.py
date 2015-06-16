import numpy as np

def make_prob(a):
    return a/a.sum()

def do_along(f, M, axis):
    M2 = np.swapaxes(M, 0, axis)
    ndim = M2.shape[0]
    return np.array([ f(M2[i, ...]) for i in range(ndim) ])
