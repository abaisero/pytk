import unittest

import pytk.decorators as decorators


class StaticTest(unittest.TestCase):
    def test_static(self):
        @decorators.static(m=2, b=-1)
        def f(x):
            return f.m * x + f.b

        self.assertEqual(f.m, 2)
        self.assertEqual(f.b, -1)

        self.assertEqual(f(-1), -3)
        self.assertEqual(f(0), -1)
        self.assertEqual(f(1), 1)
        self.assertEqual(f(2), 3)
