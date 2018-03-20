import unittest

import pytk.decorators as decorators


class Foo(object):
    def __init__(self, n):
        self.n = n

    @decorators.sentinel('n')
    def prop(self):
        return dict(n=self.n)


class Sentineltest(unittest.TestCase):
    def setUp(self):
        self.foo = Foo(10)

    def test_sentinel(self):
        self.assertEqual(self.foo.prop, dict(n=10))

        prop = self.foo.prop
        self.assertEqual(self.foo.prop, prop)
        self.assertIs(self.foo.prop, prop)

        self.foo.n += 1
        self.assertIs(self.foo.prop, self.foo.prop)
        self.assertNotEqual(self.foo.prop, prop)
        self.assertIsNot(self.foo.prop, prop)

        self.foo.n -= 1
        self.assertIs(self.foo.prop, self.foo.prop)
        self.assertEqual(self.foo.prop, prop)
        self.assertIsNot(self.foo.prop, prop)

        del self.foo.prop
        self.assertIs(self.foo.prop, self.foo.prop)
        self.assertEqual(self.foo.prop, prop)
        self.assertIsNot(self.foo.prop, prop)
