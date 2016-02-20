import unittest
import time

import pytk.decorators as decorators


class Foo(object):
    def __init__(self):
        self.n = 0

    @decorators.memoize
    def f_memoized(self, *args, **kwargs):
        return tuple(args) + tuple(kwargs.items())

    @decorators.lazyprop
    def p_lazyprop(self):
        return dict(a=1, b=2)

    @decorators.sentinel('n')
    def p_sentinel(self):
        return dict(n=self.n)

    @decorators.once_every_nth(3)
    def f_once_every_nth(self):
        self.n += 1

    @decorators.once_every_period(.1)
    def f_once_every_period(self):
        self.n += 1


class DecoratorTest(unittest.TestCase):
    def setUp(self):
        self.foo = Foo()

    def test_memoize(self):
        # Functions can be memoized
        obj1 = self.foo.f_memoized(1, 2, c=3, d=4)
        obj2 = self.foo.f_memoized(1, 2, c=3, d=5)
        obj3 = self.foo.f_memoized(1, 2, c=3, d=4)
        self.assertIsNot(obj1, obj2)
        self.assertIsNot(obj2, obj3)
        self.assertIs(obj1, obj3)

    def test_lazyprop(self):
        # Class properties can be memoized
        obj1 = self.foo.p_lazyprop
        obj2 = self.foo.p_lazyprop
        del self.foo.p_lazyprop
        obj3 = self.foo.p_lazyprop

        self.assertIs(obj1, obj2)
        self.assertIsNot(obj1, obj3)

    def test_sentinel(self):
        # Class property can be memoized conditional to other attribute values
        for i in range(10):
            self.foo.n = i
            obj1 = self.foo.p_sentinel
            self.assertIs(obj1, self.foo.p_sentinel)

            self.foo.n = i + 1
            obj2 = self.foo.p_sentinel
            self.assertIs(obj2, self.foo.p_sentinel)

            self.foo.n = i
            obj3 = self.foo.p_sentinel
            self.assertIs(obj3, self.foo.p_sentinel)

            self.assertNotEqual(obj1, obj2)
            self.assertIsNot(obj1, obj3)
            self.assertEqual(obj1, obj3)

        self.foo.n = i
        obj1 = self.foo.p_sentinel
        del self.foo.p_sentinel
        obj2 = self.foo.p_sentinel
        self.assertEqual(obj1, obj2)
        self.assertIsNot(obj1, obj2)

    def test_once_every_nth(self):
        # Function execution can be controlled discretely
        self.assertEqual(self.foo.n, 0)
        self.assertEqual(self.foo.f_once_every_nth.n, 0)
        for i in range(20):
            self.foo.f_once_every_nth()
            self.assertEqual(self.foo.n, i / self.foo.f_once_every_nth.period + 1)
            self.assertEqual(self.foo.f_once_every_nth.n, i + 1)

    def test_once_every_period(self):
        # Function execution can be controlled temporally
        self.assertEqual(self.foo.n, 0)
        self.assertEqual(self.foo.f_once_every_period.n, 0)

        sleep_hz = 20
        sleep_t = self.foo.f_once_every_period.period / sleep_hz
        for i in range(20):
            for j in range(sleep_hz):
                time.sleep(sleep_t)
                self.foo.f_once_every_period()
                self.assertEqual(self.foo.n, i + 1)
                self.assertEqual(self.foo.f_once_every_period.n, sleep_hz * i + j + 1)
