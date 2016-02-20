import unittest

import pytk.importable as importable


class Foo(object):
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j


class ImportableTest(unittest.TestCase):
    def test_load_cls(self):
        # Classes can be imported dynamically using their import_string
        cls = importable.load_cls('tests.test_importable.Foo')
        self.assertIs(cls, Foo)

    def test_load_obj(self):
        # Objects can be instantiated dynamically using their import_string
        obj1 = importable.load_obj('tests.test_importable.Foo', 0, j=1)
        obj2 = Foo(0, j=1)
        self.assertEqual(obj1, obj2)
        self.assertIsNot(obj1, obj2)

        obj1 = importable.load_obj('tests.test_importable.Foo', 0, j=1)
        obj2 = importable.load_obj('tests.test_importable.Foo', 0, j=1)
        self.assertEqual(obj1, obj2)
        self.assertIsNot(obj1, obj2)
