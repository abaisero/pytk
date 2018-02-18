import unittest

import pytk.factory as factory


class FactoryUnionTest(unittest.TestCase):
    def setUp(self):
        self.n = 10
        self.f = factory.FactoryUnion(
            a = factory.FactoryN(self.n),
            b = factory.FactoryN(self.n),
        )

    def test_factory(self):
        self.assertEqual(self.f.nitems, 20)
        self.assertEqual(len(list(self.f.values)), 20)
        self.assertEqual(len(list(self.f.items)), 20)

        for k, v in self.f.values:
            self.assertTrue(k in ('a', 'b'))
            self.assertTrue(0 <= v < 10)

        for i in range(10):
            item = self.f.item(value=('a', i))
            self.assertTrue(self.f.a.isitem(item))

            item = self.f.item(value=('b', i))
            self.assertTrue(self.f.b.isitem(item))

        for item in self.f.items:
            self.assertTrue(self.f.a.isitem(item) ^ self.f.b.isitem(item))
