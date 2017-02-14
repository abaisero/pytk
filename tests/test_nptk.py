import unittest2 as unittest

import numpy as np
import pytk.nptk as nptk


class NptkTest(unittest.TestCase):

    def test_split(self):
        origin = np.arange(10)
        sliced = nptk.split(origin, [2, 3, 4, 1])

        np.testing.assert_array_equal(sliced[0], [0, 1])
        np.testing.assert_array_equal(sliced[1], [2, 3, 4])
        np.testing.assert_array_equal(sliced[2], [5, 6, 7, 8])
        np.testing.assert_array_equal(sliced[3], [9])
