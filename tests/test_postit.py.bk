import unittest

import numpy as np
import pytk.postit as postit


class PostitTest(unittest.TestCase):
    def setUp(self):
        self.pi = postit.PostIt()
        self.pi.add('a.x.1', 2)
        self.pi.add('a.x.2', 3)
        self.pi.add('b.y.1', 4)
        self.pi.add('a.y.2', 5)

    def test_mask(self):
        # Postit correctly computes masks
        arr = np.arange(14)

        obj = self.pi.mask(arr)
        np.testing.assert_array_equal(obj, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        obj = self.pi.mask(arr, 'a')
        np.testing.assert_array_equal(obj, [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        obj = self.pi.mask(arr, 'a.x.1')
        np.testing.assert_array_equal(obj, [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        obj = self.pi.mask(arr, 'a.x.2')
        np.testing.assert_array_equal(obj, [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        obj = self.pi.mask(arr, 'a.x')
        np.testing.assert_array_equal(obj, [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        obj = self.pi.mask(arr, 'a.x', 'b')
        np.testing.assert_array_equal(obj, [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

        obj = self.pi.mask(arr, 'a', 'b')
        np.testing.assert_array_equal(obj, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    def test_filter(self):
        # Arrays can be filtered correctly
        arr = np.arange(14)

        obj = self.pi.filter(arr)
        np.testing.assert_array_equal(obj, [])
        obj = self.pi.filter(arr, full=True)
        np.testing.assert_array_equal(obj, [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        obj = self.pi.filter(arr, 'a')
        np.testing.assert_array_equal(obj, [0, 1, 2, 3, 4, 9, 10, 11, 12, 13])
        obj = self.pi.filter(arr, 'a', full=True)
        np.testing.assert_array_equal(obj, [0, 1, 2, 3, 4, 0, 0, 0, 0, 9, 10, 11, 12, 13])

        obj = self.pi.filter(arr, 'a.x.1')
        np.testing.assert_array_equal(obj, [0, 1])
        obj = self.pi.filter(arr, 'a.x.1', full=True)
        np.testing.assert_array_equal(obj, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        obj = self.pi.filter(arr, 'a.x.2')
        np.testing.assert_array_equal(obj, [2, 3, 4])
        obj = self.pi.filter(arr, 'a.x.2', full=True)
        np.testing.assert_array_equal(obj, [0, 0, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        obj = self.pi.filter(arr, 'a.x')
        np.testing.assert_array_equal(obj, [0, 1, 2, 3, 4])
        obj = self.pi.filter(arr, 'a.x', full=True)
        np.testing.assert_array_equal(obj, [0, 1, 2, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        obj = self.pi.filter(arr, 'a.x', 'b')
        np.testing.assert_array_equal(obj, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        obj = self.pi.filter(arr, 'a.x', 'b', full=True)
        np.testing.assert_array_equal(obj, [0, 1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0, 0])

        obj = self.pi.filter(arr, 'a', 'b')
        np.testing.assert_array_equal(obj, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
        obj = self.pi.filter(arr, 'a', 'b', full=True)
        np.testing.assert_array_equal(obj, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])

    def test_tags(self):
        # Tag extraction works
        obj = self.pi.subtags()
        self.assertItemsEqual(obj, ('a.x.1', 'a.x.2', 'b.y.1', 'a.y.2'))

        obj = self.pi.subtags('a')
        self.assertItemsEqual(obj, ('a.x.1', 'a.x.2', 'a.y.2'))

        obj = self.pi.subtags('a.x.1')
        self.assertItemsEqual(obj, ('a.x.1',))

        obj = self.pi.subtags('a.x.2')
        self.assertItemsEqual(obj, ('a.x.2',))

        obj = self.pi.subtags('a.x')
        self.assertItemsEqual(obj, ('a.x.1', 'a.x.2'))

        obj = self.pi.subtags('b')
        self.assertItemsEqual(obj, ('b.y.1',))

        obj = self.pi.subtags('b.y')
        self.assertItemsEqual(obj, ('b.y.1',))

        obj = self.pi.subtags('b.y.1')
        self.assertItemsEqual(obj, ('b.y.1',))
