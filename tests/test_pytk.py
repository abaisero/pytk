# import unittest2 as unittest

# from pytk.pytk import PostIt
# import numpy as np


# class PostItTest(unittest.TestCase):

#     def setUp(self):
#         self.pi = PostIt()
#         self.pi.add('a.x.1', 2)
#         self.pi.add('a.x.2', 3)
#         self.pi.add('b.y.1', 4)
#         self.pi.add('a.y.2', 5)
#         self.n = 2 + 3 + 4 + 5

#     def test_simple(self):
#         arr = np.arange(self.n)
#         np.testing.assert_array_equal(self.pi.filter(arr), np.arange(self.n))
#         np.testing.assert_array_equal(self.pi.filter(arr, 'a'), np.array([0, 1, 2, 3, 4, 9, 10, 11, 12, 13]))
#         np.testing.assert_array_equal(self.pi.filter(arr, 'a.x'), np.array([0, 1, 2, 3, 4]))
#         np.testing.assert_array_equal(self.pi.filter(arr, 'a.x.1'), np.array([0, 1]))
#         np.testing.assert_array_equal(self.pi.filter(arr, 'a.x.2'), np.array([2, 3, 4]))

# if __name__ == '__main__':
#     unittest.main()
