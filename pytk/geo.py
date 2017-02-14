from __future__ import division

import numpy as np
import numpy.linalg as la


class GeoException(Exception):
    pass


class vect(object):
    def __init__(self, xyz):
        self.xyz = xyz

    @property
    def xyz(self):
        return self.__xyz

    @xyz.setter
    def xyz(self, value):
        value = np.array(value)
        if value.shape != (3,):
            raise GeoException('Vector needs 3 values (given {})'.format(value.shape))
        self.__xyz = value

    @property
    def as_quat(self):
        return quat(np.r_[0, self.xyz])

    def __pow__(self, p):
        return np.dot(self.xyz, self.xyz) ** (p / 2)

    def __len__(self):
        return self ** 1

    def __neg__(self):
        return vect(-self.xyz)

    def __add__(self, other):
        return vect(self.xyz + other.xyz)

    def __sub__(self, other):
        return vect(self.xyz - other.xyz)
    
    def __eq__(self, other):
        return np.allclose(self.xyz, other.xyz)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return 'vect({})'.format(str(self.xyz))

    def __str__(self):
        return 'vect({})'.format(str(self.xyz))


class quat(object):
    __mulmat__ = np.array([
        [
            [ 1,  0,  0,  0],
            [ 0, -1,  0,  0],
            [ 0,  0, -1,  0],
            [ 0,  0,  0, -1],
        ],
        [
            [ 0,  1,  0,  0],
            [ 1,  0,  0,  0],
            [ 0,  0,  0,  1],
            [ 0,  0, -1,  0],
        ],
        [
            [ 0,  0,  1,  0],
            [ 0,  0,  0, -1],
            [ 1,  0,  0,  0],
            [ 0,  1,  0,  0],
        ],
        [
            [ 0,  0,  0,  1],
            [ 0,  0,  1,  0],
            [ 0, -1,  0,  0],
            [ 1,  0,  0,  0],
        ],
    ])

    def __init__(self, wxyz):
        self.wxyz = wxyz

    @property
    def wxyz(self):
        return self.__wxyz

    @wxyz.setter
    def wxyz(self, value):
        value = np.array(value)
        if value.shape != (4,):
            raise GeoException('Quaternion needs 4 values (given {})'.format(value.shape))
        self.__wxyz = value

    @property
    def w(self):
        return self.wxyz[0]

    @property
    def xyz(self):
        return self.wxyz[1:]

    @property
    def as_rquat(self):
        return rquat(self.wxyz)

    @property
    def as_vect(self):
        return vect(self.xyz)

    @property
    def conj(self):
        return type(self)(self.wxyz * [1, -1, -1, -1])

    @property
    def inv(self):
        return type(self)(self.conj.wxyz / self ** 2)

    @property
    def normal(self):
        return type(self)(self.wxyz / self ** 1)

    def __pow__(self, p):
        return np.dot(self.wxyz, self.wxyz) ** (p / 2)

    def __len__(self):
        return self ** 1

    def __neg__(self):
        return type(self)(-self.wxyz)

    def __add__(self, other):
        return quat(self.wxyz + other.wxyz)

    def __sub__(self, other):
        return quat(self.wxyz - other.wxyz)

    def __mul__(self, other):
        # NOTE returns rquat instance only if both self and other are rquats
        qclass = type(self) if isinstance(other, rquat) else quat
        wxyz = np.einsum('kij,i,j->k', quat.__mulmat__, self.wxyz, other.wxyz)
        return qclass(wxyz)

    def __div__(self, other):
        return self * other.inv

    def __eq__(self, other):
        return np.allclose(self.wxyz, other.wxyz)

    def __ne__(self, other):
        return not self == other

    def __repr__(self):
        return '{}({})'.format(type(self), str(self.wxyz))

    def __str__(self):
        return '{}({})'.format(type(self), str(self.wxyz))


class rquat(quat):
    @property
    def wxyz(self):
        return super(rquat, self).wxyz

    @wxyz.setter
    def wxyz(self, value):
        super(rquat, type(self)).wxyz.fset(self, value)
        self.__wxyz = self.wxyz / self ** 1

    @property
    def as_rquat(self):
        return self 

    def __mul__(self, other):
        try:
            return super(rquat, self).__mul__(other)
        except AttributeError:
            xyz = (self.conj * other.as_quat * self).xyz
            return vect(xyz)

    def __eq__(self, other):
        return super(rquat, self).__eq__(other) or \
                super(rquat, self).__eq__(-other)
