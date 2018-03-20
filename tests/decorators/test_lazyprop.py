import unittest

import pytk.decorators as decorators


class Foo:
    @decorators.lazyprop
    def p_lazyprop(self):
        return dict(a=1, b=2)


class LazypropTest(unittest.TestCase):
    def setUp(self):
        self.foo = Foo()

    def test_lazyprop(self):
        self.assertEqual(self.foo.p_lazyprop, dict(a=1, b=2))

        prop = self.foo.p_lazyprop
        self.assertIs(self.foo.p_lazyprop, prop)

        del self.foo.p_lazyprop
        self.assertEqual(self.foo.p_lazyprop, prop)
        self.assertIsNot(self.foo.p_lazyprop, prop)
