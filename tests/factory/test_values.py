import unittest

import pytk.factory as factory


class FactoryValuesTest(unittest.TestCase):
    def setUp(self):
        values = 'a', 'b', 'c'
        self.f = factory.FactoryValues(values)

    def test_factory(self):
        self.assertEqual(self.f.nitems, 3)
        self.assertSequenceEqual(self.f.values, 'abc')

        # item attributes
        item = f.item(0)
        self.assertEqual(item.i, 0)
        self.assertEqual(item.value, 'a')
        item = f.item(value='a')
        self.assertEqual(item.i, 0)
        self.assertEqual(item.value, 'a')
        item_copy = item.copy()
        self.assertIsNot(item, item_copy)
        self.assertEqual(item, item_copy)
        self.assertIs(item.factory, item_copy.factory)

        # item attributes
        item = self.f.item(1)
        self.assertEqual(item.i, 1)
        self.assertEqual(item.value, 'b')
        item = self.f.item(value='b')
        self.assertEqual(item.i, 1)
        self.assertEqual(item.value, 'b')
        item_copy = item.copy()
        self.assertIsNot(item, item_copy)
        self.assertEqual(item, item_copy)
        self.assertIs(item.factory, item_copy.factory)

        # item attributes
        item = self.f.item(2)
        self.assertEqual(item.i, 2)
        self.assertEqual(item.value, 'c')
        item = self.f.item(value='c')
        self.assertEqual(item.i, 2)
        self.assertEqual(item.value, 'c')
        item_copy = item.copy()
        self.assertIsNot(item, item_copy)
        self.assertEqual(item, item_copy)
        self.assertIs(item.factory, item_copy.factory)

        # item interface
        self.assertEqual(self.f.item(0, value='a'), self.f.item(0))
        self.assertEqual(self.f.item(0, value='a'), self.f.item(value='a'))
        self.assertEqual(self.f.item(1, value='b'), self.f.item(1))
        self.assertEqual(self.f.item(1, value='b'), self.f.item(value='b'))
        self.assertEqual(self.f.item(2, value='c'), self.f.item(2))
        self.assertEqual(self.f.item(2, value='c'), self.f.item(value='c'))

        #  Correct index-value pairs should not raise an exception
        try:
            self.f.item(i=0, value='a')
        except factory.FactoryException:
            self.fail('FactoryException raised by item')
        try:
            self.f.item(i=1, value='b')
        except factory.FactoryException:
            self.fail('FactoryException raised by item')
        try:
            self.f.item(i=2, value='c')
        except factory.FactoryException:
            self.fail('FactoryException raised by item')

        #  Wrong index-value pairs should raise an exception
        self.assertRaises(factory.FactoryException, self.f.item, i=0, value='b')
        self.assertRaises(factory.FactoryException, self.f.item, i=0, value='c')
        self.assertRaises(factory.FactoryException, self.f.item, i=1, value='a')
        self.assertRaises(factory.FactoryException, self.f.item, i=1, value='c')
        self.assertRaises(factory.FactoryException, self.f.item, i=2, value='a')
        self.assertRaises(factory.FactoryException, self.f.item, i=2, value='b')

    def test_item(self):
        values = 'abc'
        f1 = factory.FactoryValues(values)
        f2 = factory.FactoryValues(values)

        self.assertEqual(f1.item(1), f1.item(1))
        self.assertNotEqual(f1.item(1), f2.item(1))


class FactoryBoolTest(unittest.TestCase):
    def setUp(self):
        self.f = factory.FactoryBool()

    def test_factory(self):
        self.assertEqual(self.f.nitems, 2)
        self.assertCountEqual(self.f.values, (True, False))
