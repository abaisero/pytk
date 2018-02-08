import unittest

import pytk.factory as factory


class FactoryTest(unittest.TestCase):
    def test_choice(self):
        values = 'abc'
        f = factory.FactoryChoice(values)

        self.assertEqual(f.nitems, 3)
        self.assertSequenceEqual(f.values, values)

        for i, value in enumerate(values):
            item = f.item(i)
            self.assertEqual(item.i, i)
            self.assertEqual(item.value, value)

            item_copy = item.copy()
            self.assertIsNot(item, item_copy)
            self.assertEqual(item, item_copy)
            self.assertIs(item.factory, item_copy.factory)

        item = f.item(i=0)
        self.assertEqual(item.i, 0)
        self.assertEqual(item.value, 'a')

        item = f.item(i=1)
        self.assertEqual(item.i, 1)
        self.assertEqual(item.value, 'b')

        item = f.item(i=2)
        self.assertEqual(item.i, 2)
        self.assertEqual(item.value, 'c')


        item = f.item(value='a')
        self.assertEqual(item.i, 0)
        self.assertEqual(item.value, 'a')

        item = f.item(value='b')
        self.assertEqual(item.i, 1)
        self.assertEqual(item.value, 'b')

        item = f.item(value='c')
        self.assertEqual(item.i, 2)
        self.assertEqual(item.value, 'c')

    def test_item(self):
        values = 'abc'
        f1 = factory.FactoryChoice(values)
        f2 = factory.FactoryChoice(values)

        self.assertEqual(f1.item(1), f1.item(1))
        self.assertNotEqual(f1.item(1), f2.item(1))

    def test_bool(self):
        f = factory.FactoryBool()

        self.assertEqual(f.nitems, 2)
        self.assertCountEqual(f.values, (True, False))

    def test_subset(self):
        f = factory.FactorySubset('abc')

        self.assertEqual(f.nitems, 8)
        self.assertEqual(f.item(0).value, set())
        self.assertEqual(f.item(7).value, set('abc'))

    def test_joint(self):
        f1 = factory.FactoryBool()
        f2 = factory.FactoryChoice('abc')
        f = factory.FactoryJoint(f1=f1, f2=f2)

        self.assertEqual(f.nitems, f1.nitems * f2.nitems)
        self.assertIs(f.f1, f1)
        self.assertIs(f.f2, f2)
        self.assertEqual(f.item(0).f1, f1.item(0))
        self.assertEqual(f.item(0).f2, f2.item(0))
        self.assertEqual(f.item(f.nitems-1).f1, f1.item(f1.nitems-1))
        self.assertEqual(f.item(f.nitems-1).f2, f2.item(f2.nitems-1))

    def test_union(self):
        pass

    def test_filter(self):
        pass


# TODO repeat generic tests for multiple factories
