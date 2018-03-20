import unittest

import random
import pytk.util as util


class Foo(util.Hashable):
    def __init__(self, i):
        self.i = i

    @util.Hashable._hashable_key.getter
    def _hashable_key(self):
        return (self.i,)


class UtilTest(unittest.TestCase):
    def test_hashable(self):
        n = 10
        indl = [random.randint(-10, 10) for i in range(n)]
        inds = set(indl)

        fool = map(Foo, indl)
        foos = set(fool)

        self.assertEqual(len(foos), len(inds))
