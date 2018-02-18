import unittest

import numpy as np
import numpy.testing as npt

import pytk.geo as geo


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

        self.assertAlmostEqual((x + y    ) ** 1, np.sqrt(2))
        self.assertAlmostEqual((x     + z) ** 1, np.sqrt(2))
        self.assertAlmostEqual((    y + z) ** 1, np.sqrt(2))
        self.assertAlmostEqual((x - y    ) ** 1, np.sqrt(2))
        self.assertAlmostEqual((x     - z) ** 1, np.sqrt(2))
        self.assertAlmostEqual((    y - z) ** 1, np.sqrt(2))
        self.assertAlmostEqual((x + y    ) ** 2, 2)
        self.assertAlmostEqual((x     + z) ** 2, 2)
        self.assertAlmostEqual((    y + z) ** 2, 2)
        self.assertAlmostEqual((x - y    ) ** 2, 2)
        self.assertAlmostEqual((x     - z) ** 2, 2)
        self.assertAlmostEqual((    y - z) ** 2, 2)


class GeoQuatTest(unittest.TestCase):

    def test_shape(self):
        self.assertRaises(geo.GeoException, geo.quat, [])
        self.assertRaises(geo.GeoException, geo.quat, [0])
        self.assertRaises(geo.GeoException, geo.quat, [0, 1])
        self.assertRaises(geo.GeoException, geo.quat, [0, 1, 2])
        self.assertRaises(geo.GeoException, geo.quat, [0, 1, 2, 3, 4])
        self.assertRaises(geo.GeoException, geo.quat, [0, 1, 2, 3, 4, 5])
        self.assertRaises(geo.GeoException, geo.quat, [0, 1, 2, 3, 4, 5, 6])
        self.assertRaises(geo.GeoException, geo.quat, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_equality(self):
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertEqual(w, w)
        self.assertEqual(x, x)
        self.assertEqual(y, y)
        self.assertEqual(z, z)

        self.assertNotEqual(w, x)
        self.assertNotEqual(w, y)
        self.assertNotEqual(w, z)
        self.assertNotEqual(x, y)
        self.assertNotEqual(x, z)
        self.assertNotEqual(y, z)

    def test_add_sub(self):
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertEqual(  w + x + y    , geo.quat([1, 1, 1, 0]))
        self.assertEqual(  w + x     + z, geo.quat([1, 1, 0, 1]))
        self.assertEqual(  w +     y + z, geo.quat([1, 0, 1, 1]))
        self.assertEqual(      x + y + z, geo.quat([0, 1, 1, 1]))

        self.assertEqual(  w - x + y    , geo.quat([ 1, -1,  1,  0]))
        self.assertEqual(  w - x     + z, geo.quat([ 1, -1,  0,  1]))
        self.assertEqual(  w     - y + z, geo.quat([ 1,  0, -1,  1]))
        self.assertEqual(      x - y + z, geo.quat([ 0,  1, -1,  1]))
        self.assertEqual(- w + x - y    , geo.quat([-1,  1, -1,  0]))
        self.assertEqual(- w + x     - z, geo.quat([-1,  1,  0, -1]))
        self.assertEqual(- w     + y - z, geo.quat([-1,  0,  1, -1]))
        self.assertEqual(    - x + y - z, geo.quat([ 0, -1,  1, -1]))

    def test_neg(self):
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertEqual(w - x, w + (-x))
        self.assertEqual(w - y, w + (-y))
        self.assertEqual(w - z, w + (-z))
        self.assertEqual(x - y, x + (-y))
        self.assertEqual(x - z, x + (-z))
        self.assertEqual(y - z, y + (-z))

    def test_pow(self):
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertAlmostEqual(w ** 2, 1)
        self.assertAlmostEqual(x ** 2, 1)
        self.assertAlmostEqual(y ** 2, 1)
        self.assertAlmostEqual(z ** 2, 1)

        self.assertAlmostEqual((  w - x + y    ) ** 2, 3)
        self.assertAlmostEqual((  w - x     + z) ** 2, 3)
        self.assertAlmostEqual((  w     - y + z) ** 2, 3)
        self.assertAlmostEqual((      x - y + z) ** 2, 3)
        self.assertAlmostEqual((- w + x - y    ) ** 2, 3)
        self.assertAlmostEqual((- w + x     - z) ** 2, 3)
        self.assertAlmostEqual((- w     + y - z) ** 2, 3)
        self.assertAlmostEqual((    - x + y - z) ** 2, 3)

    def test_conj(self):
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertEqual(w + w.conj, geo.quat([2, 0, 0, 0]))
        self.assertEqual(x + x.conj, geo.quat([0, 0, 0, 0]))
        self.assertEqual(y + y.conj, geo.quat([0, 0, 0, 0]))
        self.assertEqual(z + z.conj, geo.quat([0, 0, 0, 0]))

    def test_inv(self):
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertEqual(w * w.inv, geo.quat([1, 0, 0, 0]))
        self.assertEqual(w.inv * w, geo.quat([1, 0, 0, 0]))
        self.assertEqual(x * x.inv, geo.quat([1, 0, 0, 0]))
        self.assertEqual(x.inv * x, geo.quat([1, 0, 0, 0]))
        self.assertEqual(y * y.inv, geo.quat([1, 0, 0, 0]))
        self.assertEqual(y.inv * y, geo.quat([1, 0, 0, 0]))
        self.assertEqual(z * z.inv, geo.quat([1, 0, 0, 0]))
        self.assertEqual(z.inv * z, geo.quat([1, 0, 0, 0]))

    def test_normal(self):
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertAlmostEqual((w            ).normal ** 2, 1)
        self.assertAlmostEqual((    x        ).normal ** 2, 1)
        self.assertAlmostEqual((        y    ).normal ** 2, 1)
        self.assertAlmostEqual((            z).normal ** 2, 1)
        self.assertAlmostEqual((w + x        ).normal ** 2, 1)
        self.assertAlmostEqual((w     + y    ).normal ** 2, 1)
        self.assertAlmostEqual((w         + z).normal ** 2, 1)
        self.assertAlmostEqual((    x + y    ).normal ** 2, 1)
        self.assertAlmostEqual((    x     + z).normal ** 2, 1)
        self.assertAlmostEqual((        y + z).normal ** 2, 1)
        self.assertAlmostEqual((w + x + y    ).normal ** 2, 1)
        self.assertAlmostEqual((w + x     + z).normal ** 2, 1)
        self.assertAlmostEqual((w +     y + z).normal ** 2, 1)
        self.assertAlmostEqual((    x + y + z).normal ** 2, 1)
        self.assertAlmostEqual((w + x + y + z).normal ** 2, 1)

    def test_as_rquat(self):
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertEqual((w            ).as_rquat, rquat([1, 0, 0, 0]))
        self.assertEqual((    x        ).as_rquat, rquat([0, 1, 0, 0]))
        self.assertEqual((        y    ).as_rquat, rquat([0, 0, 1, 0]))
        self.assertEqual((            z).as_rquat, rquat([0, 0, 0, 1]))
        self.assertEqual((w + x        ).as_rquat, rquat([1, 1, 0, 0]))
        self.assertEqual((w     + y    ).as_rquat, rquat([1, 0, 1, 0]))
        self.assertEqual((w         + z).as_rquat, rquat([1, 0, 0, 1]))
        self.assertEqual((    x + y    ).as_rquat, rquat([0, 1, 1, 0]))
        self.assertEqual((    x     + z).as_rquat, rquat([0, 1, 0, 1]))
        self.assertEqual((        y + z).as_rquat, rquat([0, 0, 1, 1]))
        self.assertEqual((w + x + y    ).as_rquat, rquat([1, 1, 1, 0]))
        self.assertEqual((w + x     + z).as_rquat, rquat([1, 1, 0, 1]))
        self.assertEqual((w +     y + z).as_rquat, rquat([1, 0, 1, 1]))
        self.assertEqual((    x + y + z).as_rquat, rquat([0, 1, 1, 1]))
        self.assertEqual((w + x + y + z).as_rquat, rquat([1, 1, 1, 1]))

    def test_as_vect(self):
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertEqual((w            ).as_vect, vect([0, 0, 0]))
        self.assertEqual((    x        ).as_vect, vect([1, 0, 0]))
        self.assertEqual((        y    ).as_vect, vect([0, 1, 0]))
        self.assertEqual((            z).as_vect, vect([0, 0, 1]))
        self.assertEqual((w + x        ).as_vect, vect([1, 0, 0]))
        self.assertEqual((w     + y    ).as_vect, vect([0, 1, 0]))
        self.assertEqual((w         + z).as_vect, vect([0, 0, 1]))
        self.assertEqual((    x + y    ).as_vect, vect([1, 1, 0]))
        self.assertEqual((    x     + z).as_vect, vect([1, 0, 1]))
        self.assertEqual((        y + z).as_vect, vect([0, 1, 1]))
        self.assertEqual((w + x + y    ).as_vect, vect([1, 1, 0]))
        self.assertEqual((w + x     + z).as_vect, vect([1, 0, 1]))
        self.assertEqual((w +     y + z).as_vect, vect([0, 1, 1]))
        self.assertEqual((    x + y + z).as_vect, vect([1, 1, 1]))
        self.assertEqual((w + x + y + z).as_vect, vect([1, 1, 1]))

    def test_mul(self):
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertEqual(  w * x          , geo.quat([ 0,  1,  0,  0]))
        self.assertEqual(  w     * y      , geo.quat([ 0,  0,  1,  0]))
        self.assertEqual(  w         * z  , geo.quat([ 0,  0,  0,  1]))
        self.assertEqual(      x * y      , geo.quat([ 0,  0,  0,  1]))
        self.assertEqual(      x     * z  , geo.quat([ 0,  0, -1,  0]))
        self.assertEqual(          y * z  , geo.quat([ 0,  1,  0,  0]))
        self.assertEqual(  w * x * y      , geo.quat([ 0,  0,  0,  1]))
        self.assertEqual(  w * x     * z  , geo.quat([ 0,  0, -1,  0]))
        self.assertEqual(  w     * y * z  , geo.quat([ 0,  1,  0,  0]))
        self.assertEqual(      x * y * z  , geo.quat([-1,  0,  0,  0]))
        self.assertEqual(  w * x * y * z  , geo.quat([-1,  0,  0,  0]))

        self.assertEqual( (w * x) * y * z , geo.quat([-1,  0,  0,  0]))
        self.assertEqual( w * (x * y) * z , geo.quat([-1,  0,  0,  0]))
        self.assertEqual( w * x * (y * z) , geo.quat([-1,  0,  0,  0]))

        self.assertEqual((w * x * y) * (z), geo.quat([-1,  0,  0,  0]))
        self.assertEqual((w * x) * (y * z), geo.quat([-1,  0,  0,  0]))
        self.assertEqual((w) * (x * y * z), geo.quat([-1,  0,  0,  0]))

        self.assertNotEqual(x * y, y * x)
        self.assertNotEqual(x * z, z * x)
        self.assertNotEqual(y * z, z * y)

    def test_rotate(self):
        v = geo.vect([0, 1, 2])
        w = geo.quat([1, 0, 0, 0])
        x = geo.quat([0, 1, 0, 0])
        y = geo.quat([0, 0, 1, 0])
        z = geo.quat([0, 0, 0, 1])

        self.assertEqual(  w             * v  , geo.vect([0,  1,  2]))
        self.assertEqual(      x         * v  , geo.vect([0, -1, -2]))
        self.assertEqual(          y     * v  , geo.vect([0,  1, -2]))
        self.assertEqual(              z * v  , geo.vect([0, -1,  2]))
        self.assertEqual(  w * x         * v  , geo.vect([0, -1, -2]))
        self.assertEqual(  w     * y     * v  , geo.vect([0,  1, -2]))
        self.assertEqual(  w         * z * v  , geo.vect([0, -1,  2]))
        self.assertEqual(      x * y     * v  , geo.vect([0, -1,  2]))
        self.assertEqual(      x     * z * v  , geo.vect([0,  1, -2]))
        self.assertEqual(          y * z * v  , geo.vect([0, -1, -2]))
        self.assertEqual(  w * x * y     * v  , geo.vect([0, -1,  2]))
        self.assertEqual(  w * x     * z * v  , geo.vect([0,  1, -2]))
        self.assertEqual(  w     * y * z * v  , geo.vect([0, -1, -2]))
        self.assertEqual(      x * y * z * v  , geo.vect([0,  1,  2]))
        self.assertEqual(  w * x * y * z * v  , geo.vect([0,  1,  2]))


class GeoQuatTest(unittest.TestCase):
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
