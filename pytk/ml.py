from __future__ import division

import numpy as np

def as_dist(p, minp=None, tol=1e-10):
    p[(-tol<p)&(p<tol)] = 0
    if p.min() < 0 < p.max():
        print p.min(), p.max()
        raise ValueError('Input distribution may not contain negative values.')
    p = p/p.sum()
    if minp is not None:
        minp_max = p.size**-1
        if minp > minp_max:
            raise ValueError('For array of size {}, the maximum value for pmin is {:.2f}. (actual: {:.2f})'.format(p.size, minp_max, minp))
        deficit_p = p-minp
        deficit_p[deficit_p<0] = 0
        p = minp + (1-minp*p.size)*deficit_p/deficit_p.sum()
    return p
