import unittest

import pytk.factory as factory


class FactoryTest(unittest.TestCase):
    def setUp(self):
        self.f = factory.FactorySubset('abc')

    def test_subset(self):
        self.assertEqual(self.f.nitems, 8)
        self.assertEqual(self.f.item(0).value, set())
        self.assertEqual(self.f.item(7).value, set('abc'))
