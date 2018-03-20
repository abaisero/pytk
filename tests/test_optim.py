import unittest

import pytk.optim as option


class OptimTest(unittest.TestCase):
    def test_argmax(self):
        x = [0, 1, 2, 3, 4]
        f = [1, 2, 1, 2, 1]

        amax = optim.argmax(f.__getitem__, x)
        self.assertEqual(amax, 1)

        amax = optim.argmax(f.__getitem__, x, every=True)
        self.assertCountEqual(amax, [1, 3])

        for i in range(10):
            amax = optim.argmax(f.__getitem__, x, random=True)
            self.assertTrue(amax in [1, 3])


        x = [0, 1, 2, 3, 4]
        f = [1, 2, 9, 2, 1]

        amax = optim.argmax(f.__getitem__, x)
        self.assertEqual(amax, 2)

        amax = optim.argmax(f.__getitem__, x, every=True)
        self.assertCountEqual(amax, [2,])

        for i in range(10):
            amax = optim.argmax(f.__getitem__, x, random=True)
            self.assertTrue(amax in [2,])
