import unittest

import pytk.decorators as decorators


class AsTypeTest(unittest.TestCase):
    def test_aslist(self):
        @decorators.aslist
        def f(n):
            for i in range(n):
                yield i

        self.assertEqual(f(3), [0, 1, 2])

    def test_astuple(self):
        @decorators.astuple
        def f(n):
            for i in range(n):
                yield i

        self.assertEqual(f(3), (0, 1, 2))

    def test_asdict(self):
        @decorators.asdict
        def f(n):
            for i in range(n):
                yield i, i**2

        self.assertEqual(f(3), {0: 0, 1: 1, 2: 4})
