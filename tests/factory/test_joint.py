import unittest

import pytk.factory as factory


class FactoryJointTest(unittest.TestCase):
    def setUp(self):
        self.f1 = factory.FactoryBool()
        self.f2 = factory.FactoryValues('abc')
        self.f = factory.FactoryJoint(f1=self.f1, f2=self.f2)

    def test_joint(self):
        self.assertEqual(self.f.nitems, self.f1.nitems * self.f2.nitems)
        self.assertIs(f.f1, self.f1)
        self.assertIs(f.f2, self.f2)
        self.assertEqual(f.item(0).f1, self.f1.item(0))
        self.assertEqual(f.item(0).f2, self.f2.item(0))
        self.assertEqual(f.item(f.nitems-1).f1, self.f1.item(self.f1.nitems-1))
        self.assertEqual(f.item(f.nitems-1).f2, self.f2.item(self.f2.nitems-1))
