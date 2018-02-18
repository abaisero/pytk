import unittest

import pytk.factory as factory


class FactoryFilterTest(unittest.TestCase):
    def setUp(self):
        values_ = list(range(10))
        self.f_ = factory.FactoryJoint(
            a = factory.FactoryValues(values_),
            b = factory.FactoryValues(values_),
        )

        self.f = factory.FactoryFilter(self.f_, self.filter)

    @staticmethod
    def filter(v):
        return v.a > v.b

    def test_factory(self):
        self.assertEqual(self.f.nitems, 45)

        for v in self.f.values:
            self.assertTrue(self.filter(v))

        # TODO repeat tests on other stuff (item;  values;  etc)?
