import unittest

import numpy as np
import pytk.postit as postit


class PostitTest(unittest.TestCase):
    def test_base(self):
        # Builtin types can be unpacked
        pi = postit.PostIt()
        pi.add('a.x.1', 2)
        pi.add('a.x.2', 3)
        pi.add('b.y.1', 4)
        pi.add('a.y.2', 5)
        arr = np.arange(14)

        obj = pi.filter(arr)
        np.testing.assert_array_equal(obj, np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))

        obj = pi.filter(arr, 'a')
        np.testing.assert_array_equal(obj, np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13]))

        obj = pi.filter(arr, 'a.x')
        np.testing.assert_array_equal(obj, np.array([0, 1, 2, 3, 4]))

        obj = pi.filter(arr, 'a.x.1')
        np.testing.assert_array_equal(obj, np.array([0, 1]))

        obj = pi.filter(arr, 'a.x.2')
        np.testing.assert_array_equal(obj, np.array([2, 3, 4]))
