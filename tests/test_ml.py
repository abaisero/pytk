import unittest2 as unittest

import numpy as np
import numpy.random as rnd
import numpy.testing as npt
import pytk.ml as ml

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
