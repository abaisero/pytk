import unittest2 as unittest

import numpy as np
import numpy.random as rnd
import numpy.testing as npt
import pytk.ml as ml

import autograd.numpy as anp

from operator import itemgetter


def index_order(array):
    return map(itemgetter(0), sorted(enumerate(array), key=itemgetter(1)))


class MLTest(unittest.TestCase):

    def test_as_dist(self):
        w = rnd.uniform(0, 1, 10)
        p = ml.as_dist(w)
        self.assertGreaterEqual(p.min(), 0)
        npt.assert_approx_equal(p.sum(), 1)

        w_ind = index_order(w)
        p_ind = index_order(p)
        self.assertListEqual(w_ind, p_ind)

        w = rnd.uniform(0, 1, 10)
        p = ml.as_dist(w, .1)
        self.assertGreaterEqual(p.min(), .1)
        npt.assert_approx_equal(p.sum(), 1)

        # TODO
        # this test only works partially.
        # some elements in `w` are orderd, but not in `p`.
        # w_ind = index_order(w)
        # p_ind = index_order(p)
        # self.assertListEqual(w_ind, p_ind)

        w = np.array([-1, 0, 1])
        self.assertRaises(ValueError, ml.as_dist, w)

        w = np.array([1, 1, 1])
        self.assertRaises(ValueError, ml.as_dist, w, .5)

    def test_feats(self):
        feats = ml.Feats()
        feats.postit.add('geom.pos.x')
        feats.postit.add('geom.pos.y')
        feats.postit.add('geom.pos.z')
        feats.feats = [2, 3, 4]

        self.assertEqual(feats.nobj, 1)
        self.assertEqual(feats.nfeats, 3)

        f = feats.get(('geom.pos', 'geom.pos.x'))
        np.testing.assert_array_equal(f, [2, 3, 4, 2])

        f = feats.get(('geom.pos.y', 'geom.pos.z', 'geom.pos.x'))
        np.testing.assert_array_equal(f, [3, 4, 2])

        f = feats.get(('geom.pos',))
        np.testing.assert_array_equal(f, [2, 3, 4])

        feats2 = ml.Feats()
        feats2.postit.add('geom.pos.z')
        feats2.postit.add('geom.pos.x')
        feats2.feats = feats

        f = feats2.feats
        np.testing.assert_array_equal(f, [4, 2])

    def test_feats_2(self):
        feats = ml.Feats()
        feats.postit.add('geom.pos.x')
        feats.postit.add('geom.pos.y')
        feats.postit.add('geom.pos.z')
        feats.feats = [[2, 3, 4]]

        self.assertEqual(feats.nobj, 1)
        self.assertEqual(feats.nfeats, 3)

        f = feats.get(('geom.pos', 'geom.pos.x'))
        np.testing.assert_array_equal(f, [[2, 3, 4, 2]])

        f = feats.get(('geom.pos.y', 'geom.pos.z', 'geom.pos.x'))
        np.testing.assert_array_equal(f, [[3, 4, 2]])

        f = feats.get(('geom.pos',))
        np.testing.assert_array_equal(f, [[2, 3, 4]])

        feats2 = ml.Feats()
        feats2.postit.add('geom.pos.z')
        feats2.postit.add('geom.pos.x')
        feats2.feats = feats

        f = feats2.feats
        np.testing.assert_array_equal(f, [[4, 2]])

    def test_feats_multi(self):
        feats = ml.Feats()
        feats.postit.add('geom.pos.x')
        feats.postit.add('geom.pos.y')
        feats.postit.add('geom.pos.z')
        feats.feats = [[0, 1, 2],
                       [3, 4, 5],
                       [6, 7, 8]]

        self.assertEqual(feats.nobj, 3)
        self.assertEqual(feats.nfeats, 3)

        f = feats.get(('geom.pos', 'geom.pos.x'))
        np.testing.assert_array_equal(f, [[0, 1, 2, 0],
                                          [3, 4, 5, 3],
                                          [6, 7, 8, 6]])

        f = feats.get(('geom.pos.y', 'geom.pos.z', 'geom.pos.x'))
        np.testing.assert_array_equal(f, [[1, 2, 0],
                                          [4, 5, 3],
                                          [7, 8, 6]])

        f = feats.get(('geom.pos',))
        np.testing.assert_array_equal(f, [[0, 1, 2],
                                          [3, 4, 5],
                                          [6, 7, 8]])

        feats2 = ml.Feats()
        feats2.postit.add('geom.pos.z')
        feats2.postit.add('geom.pos.x')
        feats2.feats = feats

        np.testing.assert_array_equal(feats2.feats, [[2, 0],
                                                     [5, 3],
                                                     [8, 6]])

    def test_phi(self):
        def fun(x):
            # return np.array([x[1], x[0], x[2], x[1] - x[0], x.prod()])
            return np.array([x[1], x[0], x[2], x[1] - x[0], x[0] * x[1] * x[2]])

        phi = ml.Phi(fun)
        phi.x.postit.add('geom.pos.x')
        phi.x.postit.add('geom.pos.y')
        phi.x.postit.add('geom.pos.z')

        phi.y.postit.add('geom.pos.y')
        phi.y.postit.add('geom.pos.x')
        phi.y.postit.add('geom.pos.z')
        phi.y.postit.add('geom.pos.diff.yx')
        phi.y.postit.add('geom.pos.prod')

        phi.feats = [0, 1, 2]
        np.testing.assert_array_equal(phi.feats, [1, 0, 2, 1, 0])
        np.testing.assert_array_equal(phi.jac, [[0, 1, 0],
                                                [1, 0, 0],
                                                [0, 0, 1],
                                                [-1, 1, 0],
                                                [2, 0, 0]])
        np.testing.assert_array_equal(phi.hess, [[[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 2, 1],
                                                  [2, 0, 0],
                                                  [1, 0, 0]]])

        phi.feats = [1, -1, -1]
        np.testing.assert_array_equal(phi.feats, [-1, 1, -1, -2, 1])
        np.testing.assert_array_equal(phi.jac, [[0, 1, 0],
                                                [1, 0, 0],
                                                [0, 0, 1],
                                                [-1, 1, 0],
                                                [1, -1, -1]])
        np.testing.assert_array_equal(phi.hess, [[[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, -1, -1],
                                                  [-1, 0, 1],
                                                  [-1, 1, 0]]])

        def fun(x):
            x1 = [a for a in x]
            x2 = [a * a for a in x]
            return np.array([a for a in x1 + x2])

        phi = ml.Phi(fun)
        phi.x.postit.add('geom.pos.x')
        phi.x.postit.add('geom.pos.y')
        phi.x.postit.add('geom.pos.z')

        phi.y.postit.add('geom.pos.x')
        phi.y.postit.add('geom.pos.y')
        phi.y.postit.add('geom.pos.z')
        phi.y.postit.add('geom.pos.x2')
        phi.y.postit.add('geom.pos.y2')
        phi.y.postit.add('geom.pos.z2')

        phi.feats = [0, 1, 2]
        np.testing.assert_array_equal(phi.feats, [0, 1, 2, 0, 1, 4])
        np.testing.assert_array_equal(phi.jac, [[1, 0, 0],
                                                [0, 1, 0],
                                                [0, 0, 1],
                                                [0, 0, 0],
                                                [0, 2, 0],
                                                [0, 0, 4]])
        np.testing.assert_array_equal(phi.hess, [[[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[2, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 2, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 2]]])

        phi.feats = [-1, 0, 1]
        np.testing.assert_array_equal(phi.feats, [-1, 0, 1, 1, 0, 1])
        np.testing.assert_array_equal(phi.jac, [[1, 0, 0],
                                                [0, 1, 0],
                                                [0, 0, 1],
                                                [-2, 0, 0],
                                                [0, 0, 0],
                                                [0, 0, 2]])
        np.testing.assert_array_equal(phi.hess, [[[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[2, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 2, 0],
                                                  [0, 0, 0]],
                                                 [[0, 0, 0],
                                                  [0, 0, 0],
                                                  [0, 0, 2]]])

        np.testing.assert_array_equal(phi.jac.shape, [6, 3])
        np.testing.assert_array_equal(phi.jac_shape, [6, 3])
        np.testing.assert_array_equal(phi.hess.shape, [6, 3, 3])
        np.testing.assert_array_equal(phi.hess_shape, [6, 3, 3])
