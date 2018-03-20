import unittest

import pytk.probability as probability
import numpy as np


class ProbabilityTest(unittest.TestCase):
    def setUp(self):
        self.a = np.arange(24).reshape((2, 3, 4))

    def test_normal(self):
        a = probability.normal(self.a)
        self.assertEqual(a.shape, (2, 3, 4))
        self.assertEqual(a.sum(), 1.)

    def test_conditional(self):
        a = probability.normal(self.a)
        b = probability.conditional(a, (0, 1))

        self.assertEqual(b.shape, (2, 3, 4))

    def test_marginal(self):
        a = probability.normal(self.a)
        b = probability.marginal(a, 0)

        self.assertEqual(b.shape, (3, 4))
