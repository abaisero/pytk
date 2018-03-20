import unittest

import pytk.geo as geo


class GeoRQuatTest(unittest.TestCase):
    def test_pow(self):
        w = geo.rquat([1, 0, 0, 0])
        x = geo.rquat([0, 1, 0, 0])
        y = geo.rquat([0, 0, 1, 0])
        z = geo.rquat([0, 0, 0, 1])

        self.assertAlmostEqual((w            ) ** 2, 1)
        self.assertAlmostEqual((    x        ) ** 2, 1)
        self.assertAlmostEqual((        y    ) ** 2, 1)
        self.assertAlmostEqual((            z) ** 2, 1)
        self.assertAlmostEqual((w * x        ) ** 2, 1)
        self.assertAlmostEqual((w     * y    ) ** 2, 1)
        self.assertAlmostEqual((w         * z) ** 2, 1)
        self.assertAlmostEqual((    x * y    ) ** 2, 1)
        self.assertAlmostEqual((    x     * z) ** 2, 1)
        self.assertAlmostEqual((        y * z) ** 2, 1)
        self.assertAlmostEqual((w * x * y    ) ** 2, 1)
        self.assertAlmostEqual((w * x     * z) ** 2, 1)
        self.assertAlmostEqual((w *     y * z) ** 2, 1)
        self.assertAlmostEqual((    x * y * z) ** 2, 1)
        self.assertAlmostEqual((w * x * y * z) ** 2, 1)
