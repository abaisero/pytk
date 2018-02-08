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

    def test_argmax(self):
        x = [0, 1, 2, 3, 4]
        f = [1, 2, 1, 2, 1]

        amax = util.argmax(f.__getitem__, x)
        self.assertEqual(amax, 1)

        amax = util.argmax(f.__getitem__, x, all_=True)
        self.assertCountEqual(amax, [1, 3])

        for i in range(10):
            amax = util.argmax(f.__getitem__, x, rnd_=True)
            self.assertTrue(amax in [1, 3])


        x = [0, 1, 2, 3, 4]
        f = [1, 2, 9, 2, 1]

        amax = util.argmax(f.__getitem__, x)
        self.assertEqual(amax, 2)

        amax = util.argmax(f.__getitem__, x, all_=True)
        self.assertCountEqual(amax, [2,])

        for i in range(10):
            amax = util.argmax(f.__getitem__, x, rnd_=True)
            self.assertTrue(amax in [2,])
