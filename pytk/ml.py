from __future__ import division

import numpy as np


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


class Feats(object):
    def __init__(self):
        self._attrs = {}
        self._feats = None

    # def _add(self, fname, fnum=None):

    #     def getter(self):
    #         return

    #     def setter(self, value):
    #         self. set something

    #     prop = property(getter, setter)
    #     setattr(self.__class__, fname, prop)
    #     # self._attrs[fname] = None

    @property
    def feats(self):
        return self._feats

    @feats.setter
    def feats(self, value):
        self._feats = np.atleast_2d(value)
        # print self._feats

    def __getattr__(self, attr):
        print 'getattr {}'.format(attr)
        if attr is 'feats':
            return None
        if attr.startswith('_') or attr not in self._attrs:
            raise AttributeError('Feats object has no feature {}.'.format(attr))
        return self._attrs[attr]

    def __setattr__(self, attr, value):
        print 'setattr {} {}'.format(attr, value)
        # if hasattr(self, 'feats') and hasattr(self.feats, attr)
        # if attr == '_feats':
        #     raise AttributeError
        if attr.startswith('_'):
            self.__dict__[attr] = value
        elif attr not in self._attrs:
            raise AttributeError('Feats object has no feature {}.'.format(attr))
        self._attrs[attr] = value


class Test(object):
    def __init__(self):
        self._prop = None

    @property
    def prop(self):
        return self._prop

    @prop.setter
    def prop(self, value):
        self._prop = value

    def __getattr__(self, attr):
        # print '__getattr__', attr
        pass

    def __setattr__(self, attr, value):
        # print '__setattr__', attr, value
        if attr is 'prop':
            Test.prop.fset(self, value)
        if attr.startswith('_'):
            self.__dict__[attr] = value


if __name__ == '__main__':
    test = Test()

    print '---'
    print 'prop', test.prop
    print '---'
    test.prop = 2
    print 'prop', test.prop
    print '---'

    print test.__class__

    # feats = Feats()
    # feats._add('pos', 2)
    # feats._add('red', 1)
    # feats._add('blue', 1)

    # print '---'
    # feats.pos = 1
    # print '---'
    # feats.pos = 2
    # print '---'
    # print feats.pos
    # print '---'
    # print feats.feats
    # print '---'
    # feats.feats = [1, 2]
    # print feats.feats
