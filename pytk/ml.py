from __future__ import division

from collections import OrderedDict

import numpy as np
import pytk.postit
import pytk.nptk as nptk
import pytk.pack as pack
import pytk.itt as itt
from pytk.decorators import setprop

from autograd import jacobian, hessian

class MLException(Exception):
    pass


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


# TODO Feats should not be indexed directly by index, but by object names!!! Anything can be a key!

class Feats(pack.Serializable):

    @property
    def _key(self):
        return (self.postit, self.feats)

    def _encode(self):
        return dict(postit=self.postit, feats=self.feats)

    @classmethod
    def _decode(cls, data):
        keys = ['postit', 'feats']
        postit, feats = (data[key] for key in keys)
        obj = cls(postit)
        obj.feats = feats
        return obj

    def __eq__(self, other):
        return isinstance(self, type(other)) \
            and self.postit == other.postit \
            and np.array_equal(self.feats, other.feats)

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
        if self._feats is not None:
            return self._feats
        return np.empty((0, self.postit.n))

    @property
    def feats_2d(self):
        if self.feats is not None:
            return np.atleast_2d(self.feats)
        # return np.empty((0, self.postit.n))

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

    def __str__(self):
        return '{}\n{}'.format(self.postit, self.feats)

class Feats(object):
    def __init__(self, postit=None, _feats=None, tagsview=False):
        if postit is None:
            postit = pytk.postit.PostIt()
        self.postit = postit
        if _feats is None:
            _feats = OrderedDict()
        self._feats = _feats
        self.tagsview = tagsview

    def clear(self):
        self._feats.clear()

    def rename(self, oldkey, newkey, inplace):
        _feats = self._feats.copy()
        try:
            for oldk, newk in zip(oldkey, newkey):
                _feats[newk] = self._feats[oldk]
        except TypeError:
            _feats[newkey] = self._feats[oldkey]
        if inplace:
            self._feats = _feats
            return self
        return Feats(postit=self.postit, _feats=_feats)

    @property
    def objects(self):
        return self._feats.keys()

    @property
    def nobj(self):
        return len(self._feats)

    @property
    def feats(self):
        return np.array(self._feats.values()).squeeze()

    @property
    def feats_2d(self):
        return np.atleast_2d(self.feats)

    @property
    def nfeats(self):
        return self.postit.n

    # # TODO what is this?????
    # def mask(self, obj=None, tags=None):
    #     objm = np.ones(self.nobj, dtype=bool)
    #     if obj is not None:
    #         objm = np.zeros(self.nobj, dtype=bool)
    #         obji = self._feats.keys().index(obj)
    #         objm[obji] = True

    #     tagm = np.ones(self.nfeats, dtype=bool)
    #     if tags is not None:
    #         tagm = self.postit.mask(*tags)

    #     return np.outer(objm, tagm)

    # # TODO obsolete.. use __getitem__
    # def get(self, obj=None, tags=None):
    #     # TODO check what happens if nobj is 0
    #     # TODO check what happens is obj doesn't exist
    #     objm = np.ones(self.nobj, dtype=bool)
    #     if obj is not None:
    #         objm = np.zeros(self.nobj, dtype=bool)
    #         obji = self._feats.keys().index(obj)
    #         objm[obji] = True

    #     tagm = np.ones(self.nfeats, dtype=bool)
    #     if tags is not None:
    #         tagm = self.postit.mask(*tags)

    #     if objm.size == 0:
    #         return np.empty((0, tagm.sum()))
    #     return self.feats[objm][:, tagm]

    def __getitem__(self, keys):
        if not isinstance(keys, tuple) and not isinstance(keys, slice):
            keys = (keys,)
        if self.tagsview:
            if isinstance(keys, slice) and keys == slice(None):
                tagm = np.ones(self.nfeats, dtype=bool)
            else:
                tagm = self.postit.mask(*keys)
            return self.feats_2d[:, tagm].squeeze()
        else:
            if isinstance(keys, slice) and keys == slice(None):
                _feats = self._feats
            else:
                _feats = OrderedDict((k, self._feats[k]) for k in keys)
            return Feats(postit=self.postit, _feats=_feats, tagsview=True)

    def __setitem__(self, keys, values):
        # TODO how to do this??
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = np.atleast_2d(values)
        if self.tagsview:
            # keys represent tags.. I should only overwrite the values at those tags..
            pass
        else:
            self._feats.update(zip(keys, values))

        # # even here, it depends on whether we are in tagsview or not?!
        # value = np.atleast_2d(value)
        # assert value.shape[1] == self.nfeats
        # if not isinstance(key, tuple):
        #     key = (key,)
        # for k, v in zip(key, value):
        #     self._feats[k] = v

    # def __delitem__(self, key):
    #     if not isinstance(key, tuple):
    #         key = (key,)
    #     for k in key:
    #         del self._feats[k]

    # TODO what was this about??
    # def iterfeats(self):
    #     return self._feats.iteritems()

    def __str__(self):
        return '{}\n{}'.format(self.postit, self.feats)

class Params(pack.Serializable):
    @property
    def _key(self):
        return (self.postit, self.params)

    def _encode(self):
        return dict(postit=self.postit, params=self.params)

    @classmethod
    def _decode(cls, data):
        keys = ['postit', 'params']
        postit, params = (data[key] for key in keys)
        obj = cls(postit)
        obj.params = params
        return obj

    def __eq__(self, other):
        return isinstance(self, type(other)) \
            and self.postit == other.postit \
            and np.array_equal(self.feats, other.feats)

    def __init__(self, postit=None):
        if postit is None:
            postit = pytk.postit.PostIt()
        self.postit = postit
        self._params = None

    @property
    def nparams(self):
        return self.params.size

    @property
    def params(self):
        if self._params is not None:
            return self._params
        return np.empty(self.postit.n)

    @params.setter
    def params(self, value):
        value = np.array(value, dtype=float)
        nparams = value.size
        if nparams != self.postit.n:
            raise Exception('nfeats ({}) and postit.n ({}) are incongruent'.format(nparams, self.postit.n))
        self._params = value

    def get(self, *tags, **kwargs):
        return self.postit.filter(self.params, *tags, **kwargs)


class Phi(pack.Serializable):

    @property
    def _key(self):
        # TODO can't really compare functions... but can compare feats!
        return (self.x, self.y)

    def _encode(self):
        try:
            import marshal
            fcode = marshal.dumps(self.f.func_code)
        except AttributeError:
            fcode = None
        return dict(fcode=fcode, x=self.x, y=self.y)

    @classmethod
    def _decode(cls, data):
        keys = ['fcode', 'x', 'y']
        fcode, x, y = (data[key] for key in keys)

        import marshal
        code = marshal.loads(fcode)
        import types
        f = types.FunctionType(code, globals(), 'f')

        obj = cls(f)
        obj.x = x
        obj.y = y
        return obj

    def __init__(self, f=None):
        if f is None:
            def f(x):
                return x
        self.f = f
        self.j = jacobian(f)
        self.h = hessian(f)

        self.x = Feats()
        self.y = Feats()

    @property
    def objects(self):
        return self.x.objects

    @property
    def nobj(self):
        assert self.x.nobj == self.y.nobj
        return self.x.nobj

    @setter
    def xfeats(self, value):
        """ Set the domain features (self.x.feats) and update the corresponding range features (self.y.feats)"""
        self.x.feats = value
        if self.x.feats.ndim == 1:
            self.y.feats = self.f(self.x.feats)
        else:
            feats = map(self.f, self.x.feats)
            self.y.feats = np.array(feats).reshape((-1, self.y.nfeats))

    @property
    def J(self):
        if self.x.feats.ndim == 1:
            return self.j(self.x.feats)
        return np.array(map(self.j, self.x.feats))

    @property
    def H(self):
        if self.x.feats.ndim == 1:
            return self.h(self.x.feats)
        return np.array(map(self.h, self.x.feats))

    @property
    def J_shape(self):
        return (self.y.nfeats, self.x.nfeats)

    @property
    def H_shape(self):
        return (self.y.nfeats, self.x.nfeats, self.x.nfeats)

    def __str__(self):
        return '(Phi {} -> {})'.format(self.x.nfeats, self.y.nfeats)


# TODO create mapping class......
class Phi(object):
    def __init__(self, f=None):
        if f is None:
            def f(x):
                return x

        self.f_raw = f
        self.j_raw = jacobian(f)
        self.h_raw = hessian(f)

        self.x = Feats()
        self.y = Feats()

    def f(self, X, _2d=True):
        """ always returns as 2d ndarray. """
        if _2d:
            X_2d = X.reshape((-1, self.x.nfeats))
            Y_2d = np.array(map(self.f_raw, X_2d))
            return Y_2d
        return self.f_raw(X)

    def j(self, X, _2d=True):
        """ always returns as 2d ndarray. """
        if _2d:
            X_2d = X.reshape(-1, self.x.nfeats)
            Y_2d = np.array(map(self.j_raw, X_2d))
            # Y_2d = Y_2d.reshape(-1, self.x.nfeats)
            return Y_2d
        return self.j_raw(X)

    @property
    def objects(self):
        return self.x.objects

    @objects.setter
    def objects(self, value):
        self.x.objects = value
        self.y.objects = value

    @property
    def nobj(self):
        assert self.x.nobj == self.y.nobj
        return self.x.nobj

    def rename(self, oldkey, newkey, inplace):
        self.x.rename(oldkey, newkey, inplace)
        self.y.rename(oldkey, newkey, inplace)

    def __setitem__(self, keys, value):
        self.x[keys] = value
        self.y[keys] = self.f(value)

    # TODO feats has changed.. no more 'feats' property, and always 2D
    @property
    def J(self):
        if self.x.feats.ndim == 1:
            return self.j(self.x.feats)
        return np.array(map(self.j, self.x.feats))

    @property
    def H(self):
        if self.x.feats.ndim == 1:
            return self.h(self.x.feats)
        return np.array(map(self.h, self.x.feats))

    @property
    def J_shape(self):
        return (self.y.nfeats, self.x.nfeats)

    @property
    def H_shape(self):
        return (self.y.nfeats, self.x.nfeats, self.x.nfeats)

    def __str__(self):
        return '(Phi {} -> {})'.format(self.x.nfeats, self.y.nfeats)


class Feats(object):
    def __init__(self, base=None, okeys=None):
        self.base = base
        self.okeys = okeys

        self._postit = None
        self._farray = None
        self._objects = None

    @property
    def lenobj(self):
        return len(self._objects)

    @property
    def nobj(self):
        return len(self.objects)

    @property
    def nfeats(self):
        return self.postit.n

    @property
    def postit(self):
        if self._postit is None:
            if self.base is None:
                self._postit = pytk.postit.PostIt()
            else:
                self._postit = self.base.postit
        return self._postit

    @property
    def farray(self):
        if self._farray is None:
            if self.base is None:
                self._farray = np.empty((self.nobj, self.nfeats))
            else:
                self._farray = self.base.farray
        return self._farray

    @property
    def objects(self):
        if self._objects is None:
            if self.base is None:
                self._objects = []
            else:
                self._objects = self.base._objects
                # self._objects = list(self.base.objects)
        return tuple(filter(lambda o: o is not None, self._objects))
        return tuple(self._objects)

    @objects.setter
    def objects(self, value):
        if self.okeys is None:
            self._objects = list(value)
        else:
            self.objects  # initializing _objects
            map(self._objects.__setitem__, self.okeys, value)
            # for okey, v in zip(self.okeys, value):
            #     self._objects[okey] = v

    def alias(self, oldkeys, newkeys):
        alias = Feats(self)
        alias.objects = self.nobj * [None]
        okeys = map(self.objects.index, oldkeys)
        map(alias._objects.__setitem__, okeys, newkeys)
        return alias

    def __getitem__(self, keys):
        if not isinstance(keys, (tuple, slice)):
            keys = (keys,)
        if self.okeys is None:
            if keys == slice(None):
                okeys = range(self.nobj)
            else:
                okeys = map(self._objects.index, keys)
            return Feats(self, okeys=okeys)
        else:
            okeys = np.array(self.okeys).reshape(-1, 1)
            if keys == slice(None):
                fkeys = np.arange(self.nfeats)
            else:
                fkeys = self.postit.imask(*keys)
            return self.farray[okeys, fkeys]

    def __setitem__(self, keys, value):
        if not isinstance(keys, (tuple, slice)):
            keys = (keys,)
        if self.okeys is None:
            if keys == slice(None):
                okeys = np.arange(self.nobj)
            else:
                newkeys = filter(lambda k: k not in self.objects, keys)
                if newkeys:
                    new_farray = np.empty((len(newkeys), self.nfeats))
                    self._farray = np.vstack((self.farray, new_farray))
                    self._objects.extend(newkeys)
                # okeys = np.array([i for i, obj in enumerate(self.objects) if obj in keys])
                okeys = np.in1d(self.objects, keys).nonzero()[0]
            self.farray[okeys, :] = value
        else:
            # TODO feats[1, 2][:] = ... still doesn't work, if 1 and 2 don't exist yet
            # but feats[1, 2] = ... does
            okeys = np.array(self.okeys).reshape(-1, 1)
            self.farray[okeys, :] = value
            # self._farray[okeys, :] = value

    def __str__(self):
        _str  = 'Feats ({})'.format(self.postit.tags)
        for obj in self.objects:
            _str += '\n{}: {}'.format(obj, self[obj][:].ravel())
        return _str
