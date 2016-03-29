from __future__ import division

import numpy as np
import pytk.postit
import pytk.nptk as nptk
from pytk.decorators import sentinel


def as_dist(p, minp=None, tol=1e-3):
    p[(-tol < p) & (p < tol)] = 0
    if p.min() < 0 < p.max():
        raise ValueError('Input distribution may not contain negative values.')
    p = p / p.sum()
    if minp is not None:
        minp_max = p.size**-1
        if minp > minp_max:
            raise ValueError('For array of size {}, the maximum value for pmin is {:.2f}. (actual: {:.2f})'.format(p.size, minp_max, minp))
        deficit_p = p - minp
        deficit_p[deficit_p < 0] = 0
        p = minp + (1 - minp * p.size) * deficit_p / deficit_p.sum()
    return p


# class Phi(object):
#     def __init__(self, postit=None):
#         if postit is None:
#             postit = pytk.postit.PostIt()
#         self.postit = postit
#         self.feats = None

#     @property
#     def nobj(self):
#         if self.feats.ndim == 1:
#             return 1
#         return self.feats.shape[0]

#     @property
#     def nfeats(self):
#         if self.feats.ndim == 1:
#             return self.feats.size
#         return self.feats.shape[1]

#     def get_feats(self, tags=None, poly=None):
#         if tags is None:
#             tags = (None,)
#         masks = map(self.postit.mask, tags)
#         idxs = map(np.flatnonzero, masks)
#         feats = nptk.stack([self.feats.take(idx, axis=-1) for idx in idxs])
#         try:
#             feats = np.squeeze(feats, axis=0)
#         except ValueError:
#             pass
#         return feats

#     def set_feats(self, value):
#         if isinstance(value, Phi):
#             tags = self.postit.subtags()
#             value = value.get_feats(tags)
#         value = np.atleast_1d(value)
#         # no, nf = value.shape
#         nf = value.shape[-1]
#         if nf != self.postit.n:
#             raise Exception('nf ({}) and postit.n ({}) are incongruent'.format(nf, self.postit.n))
#         value.reshape
#         self.feats = value


# class Feats(object):
#     def __init__(self):
#         self._attrs = {}
#         self._feats = None

#     # def _add(self, fname, fnum=None):

#     #     def getter(self):
#     #         return

#     #     def setter(self, value):
#     #         self. set something

#     #     prop = property(getter, setter)
#     #     setattr(self.__class__, fname, prop)
#     #     # self._attrs[fname] = None

#     @property
#     def feats(self):
#         return self._feats

#     @feats.setter
#     def feats(self, value):
#         self._feats = np.atleast_2d(value)
#         # print self._feats

#     def __getattr__(self, attr):
#         print 'getattr {}'.format(attr)
#         if attr is 'feats':
#             return None
#         if attr.startswith('_') or attr not in self._attrs:
#             raise AttributeError('Feats object has no feature {}.'.format(attr))
#         return self._attrs[attr]

#     def __setattr__(self, attr, value):
#         print 'setattr {} {}'.format(attr, value)
#         # if hasattr(self, 'feats') and hasattr(self.feats, attr)
#         # if attr == '_feats':
#         #     raise AttributeError
#         if attr.startswith('_'):
#             self.__dict__[attr] = value
#         elif attr not in self._attrs:
#             raise AttributeError('Feats object has no feature {}.'.format(attr))
#         self._attrs[attr] = value


# class Test(object):
#     def __init__(self):
#         self._prop = None

#     @property
#     def prop(self):
#         return self._prop

#     @prop.setter
#     def prop(self, value):
#         self._prop = value

#     def __getattr__(self, attr):
#         # print '__getattr__', attr
#         pass

#     def __setattr__(self, attr, value):
#         # print '__setattr__', attr, value
#         if attr is 'prop':
#             Test.prop.fset(self, value)
#         if attr.startswith('_'):
#             self.__dict__[attr] = value


# if __name__ == '__main__':
#     test = Test()

#     print '---'
#     print 'prop', test.prop
#     print '---'
#     test.prop = 2
#     print 'prop', test.prop
#     print '---'

#     print test.__class__

#     # feats = Feats()
#     # feats._add('pos', 2)
#     # feats._add('red', 1)
#     # feats._add('blue', 1)

#     # print '---'
#     # feats.pos = 1
#     # print '---'
#     # feats.pos = 2
#     # print '---'
#     # print feats.pos
#     # print '---'
#     # print feats.feats
#     # print '---'
#     # feats.feats = [1, 2]
#     # print feats.feats


class Feats(object):
    def __init__(self, postit=None):
        if postit is None:
            postit = pytk.postit.PostIt()
        self.postit = postit
        self._feats = None

    @property
    def nobj(self):
        return self.feats_2d.shape[0]

    @property
    def nfeats(self):
        return self.feats_2d.shape[1]

    @property
    def feats(self):
        return self._feats

    @property
    def feats_2d(self):
        if self.feats is not None:
            return np.atleast_2d(self.feats)

    @feats.setter
    def feats(self, value):
        if isinstance(value, Feats):
            tags = self.postit.subtags()
            value = value.get(tags)

        value = np.array(value, dtype=float)
        nfeats = value.size if value.ndim == 1 else value.shape[1]
        if nfeats != self.postit.n:
            raise Exception('nfeats ({}) and postit.n ({}) are incongruent'.format(nfeats, self.postit.n))
        self._feats = value

    def get(self, tags=None):
        if tags is None:
            tags = (None,)
        masks = map(self.postit.mask, tags)
        idxs = map(np.flatnonzero, masks)
        value = nptk.stack([self.feats_2d.take(idx, axis=-1) for idx in idxs])
        return value.squeeze() if self.feats.ndim == 1 else value.reshape((self.nobj, -1))

    def mask_obj(self, obji):
        mask = np.zeros(self.feats_2d.shape, dtype=bool)
        mask[obji, :] = True
        return mask.ravel()

    def mask_tags(self, tags):
        mask = np.zeros(self.feats_2d.shape, dtype=bool)
        for tag in tags:
            m = self.postit.mask(tag)
            idx = np.flatnonzero(m)
            mask[:, idx] = True
        return mask.ravel()

from autograd import jacobian, hessian


class Phi(object):
    def __init__(self, fun=None):
        if fun is None:
            def fun(x):
                return x
        self.fun = fun
        self.fun_jac = jacobian(fun)
        self.fun_hess = hessian(fun)

        self.x = Feats()
        self.y = Feats()

    @property
    def feats(self):
        return self.y.feats

    @feats.setter
    def feats(self, value):
        self.x.feats = value
        if self.x.feats.ndim == 1:
            self.y.feats = self.fun(self.x.feats)
        else:
            self.y.feats = map(self.fun, self.x.feats)

    @property
    def jac(self):
        if self.x.feats.ndim == 1:
            return self.fun_jac(self.x.feats)
        return np.array(map(self.fun_jac, self.x.feats))

    @property
    def hess(self):
        if self.x.feats.ndim == 1:
            return self.fun_hess(self.x.feats)
        return np.array(map(self.fun_hess, self.x.feats))

    @property
    def jac_shape(self):
        return (self.y.nfeats, self.x.nfeats)

    @property
    def hess_shape(self):
        return (self.y.nfeats, self.x.nfeats, self.x.nfeats)
