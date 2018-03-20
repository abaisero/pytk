import unittest

import pytk.geo as geo
from math import sqrt

sqrt2 = sqrt(2)


class GeoVectTest(unittest.TestCase):

    def test_shape(self):
        self.assertRaises(geo.GeoException, geo.vect, [])
        self.assertRaises(geo.GeoException, geo.vect, [0])
        self.assertRaises(geo.GeoException, geo.vect, [0, 1])
        self.assertRaises(geo.GeoException, geo.vect, [0, 1, 2, 3])
        self.assertRaises(geo.GeoException, geo.vect, [0, 1, 2, 3, 4])
        self.assertRaises(geo.GeoException, geo.vect, [0, 1, 2, 3, 4, 5])
        self.assertRaises(geo.GeoException, geo.vect, [0, 1, 2, 3, 4, 5, 6])
        self.assertRaises(geo.GeoException, geo.vect, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_equality(self):
        x = geo.vect([1, 0, 0])
        y = geo.vect([0, 1, 0])
        z = geo.vect([0, 0, 1])

        self.assertEqual(x, x)
        self.assertEqual(y, y)
        self.assertEqual(z, z)

        self.assertNotEqual(x, y)
        self.assertNotEqual(x, z)
        self.assertNotEqual(y, z)

    def test_add_sub(self):
        x = geo.vect([1, 0, 0])
        y = geo.vect([0, 1, 0])
        z = geo.vect([0, 0, 1])

        self.assertEqual(x + y    , geo.vect([1,  1,  0]))
        self.assertEqual(x     + z, geo.vect([1,  0,  1]))
        self.assertEqual(    y + z, geo.vect([0,  1,  1]))
        self.assertEqual(x - y    , geo.vect([1, -1,  0]))
        self.assertEqual(x     - z, geo.vect([1,  0, -1]))
        self.assertEqual(    y - z, geo.vect([0,  1, -1]))

    def test_neg(self):
        x = geo.vect([1, 0, 0])
        y = geo.vect([0, 1, 0])
        z = geo.vect([0, 0, 1])

        self.assertEqual(x - y, x + (-y))
        self.assertEqual(x - z, x + (-z))
        self.assertEqual(y - z, y + (-z))

    def test_pow(self):
        x = geo.vect([1, 0, 0])
        y = geo.vect([0, 1, 0])
        z = geo.vect([0, 0, 1])

        self.assertAlmostEqual(x ** 1, 1)
        self.assertAlmostEqual(y ** 1, 1)
        self.assertAlmostEqual(z ** 1, 1)
        self.assertAlmostEqual(x ** 2, 1)
        self.assertAlmostEqual(y ** 2, 1)
        self.assertAlmostEqual(z ** 2, 1)

        self.assertAlmostEqual((x + y    ) ** 1, sqrt2)
        self.assertAlmostEqual((x     + z) ** 1, sqrt2)
        self.assertAlmostEqual((    y + z) ** 1, sqrt2)
        self.assertAlmostEqual((x - y    ) ** 1, sqrt2)
        self.assertAlmostEqual((x     - z) ** 1, sqrt2)
        self.assertAlmostEqual((    y - z) ** 1, sqrt2)
        self.assertAlmostEqual((x + y    ) ** 2, 2)
        self.assertAlmostEqual((x     + z) ** 2, 2)
        self.assertAlmostEqual((    y + z) ** 2, 2)
        self.assertAlmostEqual((x - y    ) ** 2, 2)
        self.assertAlmostEqual((x     - z) ** 2, 2)
        self.assertAlmostEqual((    y - z) ** 2, 2)
        self.assertAlmostEqual((x + y    ) ** 3, 2 * sqrt2)
        self.assertAlmostEqual((x     + z) ** 3, 2 * sqrt2)
        self.assertAlmostEqual((    y + z) ** 3, 2 * sqrt2)
        self.assertAlmostEqual((x - y    ) ** 3, 2 * sqrt2)
        self.assertAlmostEqual((x     - z) ** 3, 2 * sqrt2)
        self.assertAlmostEqual((    y - z) ** 3, 2 * sqrt2)
