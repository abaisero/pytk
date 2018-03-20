import unittest
import time

import pytk.decorators as decorators


class Foo(object):
    def __init__(self):
        self.n = 0

    @decorators.once_every(3)
    def f_once_every(self):
        self.n += 1

    @decorators.once_every_timer(.1)
    def f_once_every_timer(self):
        self.n += 1


class OnceEveryTest(unittest.TestCase):
    def setUp(self):
        self.foo = Foo()

    def test_once_every(self):
        self.assertEqual(self.foo.n, 0)
        self.assertEqual(self.foo.f_once_every.ncalls, 0)
        self.assertEqual(self.foo.f_once_every.ncalls_actual, 0)
        self.assertEqual(self.foo.f_once_every.ncalls_filtered, 0)

        for i in range(20):
            self.foo.f_once_every()
            self.assertEqual(self.foo.n, i//3 + 1)
            self.assertEqual(self.foo.f_once_every.ncalls, i + 1)
            self.assertEqual(self.foo.f_once_every.ncalls_actual, i//3 + 1)
            self.assertEqual(self.foo.f_once_every.ncalls_filtered, i - i//3)

    def test_once_every_timer(self):
        self.assertEqual(self.foo.n, 0)
        self.assertEqual(self.foo.f_once_every_timer.ncalls, 0)
        self.assertEqual(self.foo.f_once_every_timer.ncalls_actual, 0)
        self.assertEqual(self.foo.f_once_every_timer.ncalls_filtered, 0)

        self.foo.f_once_every_timer()
        self.foo.f_once_every_timer()
        self.assertEqual(self.foo.n, 1)
        self.assertEqual(self.foo.f_once_every_timer.ncalls, 2)
        self.assertEqual(self.foo.f_once_every_timer.ncalls_actual, 1)
        self.assertEqual(self.foo.f_once_every_timer.ncalls_filtered, 1)

        time.sleep(.1)
        self.foo.f_once_every_timer()
        self.foo.f_once_every_timer()
        self.foo.f_once_every_timer()
        self.assertEqual(self.foo.n, 2)
        self.assertEqual(self.foo.f_once_every_timer.ncalls, 5)
        self.assertEqual(self.foo.f_once_every_timer.ncalls_actual, 2)
        self.assertEqual(self.foo.f_once_every_timer.ncalls_filtered, 3)

        time.sleep(.1)
        self.foo.f_once_every_timer()
        self.assertEqual(self.foo.n, 3)
        self.assertEqual(self.foo.f_once_every_timer.ncalls, 6)
        self.assertEqual(self.foo.f_once_every_timer.ncalls_actual, 3)
        self.assertEqual(self.foo.f_once_every_timer.ncalls_filtered, 3)
