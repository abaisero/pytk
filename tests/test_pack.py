import unittest

import numpy as np
import pytk.pack as pack


class Foo_Generic(pack.Serializable):
    def __init__(self, **kwargs):
        self.contents = kwargs

    def __str__(self):
        """ Only for debugging the test """
        contents_str = ['{}={}'.format(k, v) for k, v in self.contents.iteritems()]
        return '(Foo_Generic ' + ', '.join(contents_str) + ')'

    def __eq__(self, other):
        """ Only required for testing """
        return self.contents == other.contents

    def _encode(self):
        """ Returns a dictionary with all that is necessary for new object instantiation """
        return self.contents

    @classmethod
    def _decode(cls, data):
        """ Reconstructs object instance from data dictionary """
        return cls(**data)


class Foo_Special(pack.Serializable):
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def __str__(self):
        return '(Foo_Special i={}, j={})'.format(self.i, self.j)

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j

    def _encode(self):
        """ Returns a dictionary with necessary for new object instantiation """
        return dict(i=self.i, j=self.j)

    @classmethod
    def _decode(cls, data):
        """ Reconstructs object instance from data dictionary """
        i, j = data['i'], data['j']
        return cls(i, j)


def pack_unpack(obj):
    packed = pack.packb(obj)
    return pack.unpackb(packed)


class PackTest(unittest.TestCase):
    def test_builtin_types(self):
        # Builtin types can be unpacked
        original = 1
        restored = pack_unpack(original)
        self.assertEqual(original, restored)

        original = 1.
        restored = pack_unpack(original)
        self.assertEqual(original, restored)

        original = [1, 2, 3]
        restored = pack_unpack(original)
        self.assertEqual(original, restored)

        # # TODO
        # # tuples are by default unpacked as lists
        # original = (1, 2, 3)
        # restored = pack_unpack(original)
        # self.assertEqual(original, restored)

        original = dict(a=1, b=2)
        restored = pack_unpack(original)
        self.assertEqual(original, restored)

        # # TODO
        # # msgpack can't unpack sets
        # original = set([1, 2, 3])
        # restored = pack_unpack(original)
        # self.assertEqual(original, restored)

    def test_ndarray(self):
        # Numpy arrays can be unpacked
        original = np.arange(12)
        restored = pack_unpack(original)
        np.testing.assert_array_equal(original, restored)

    def test_custom_types(self):
        # Serializable objects can be unpacked
        original = Foo_Generic(i=1, j=2, k=3)
        restored = pack_unpack(original)
        self.assertEqual(original, restored)

        original = Foo_Special(1, 2)
        restored = pack_unpack(original)
        self.assertEqual(original, restored)

    def test_mixed_types(self):
        # Objects can contain other objects
        original = [Foo_Generic(i=1, j=2, k=3), Foo_Special(4, 5), [6, 7]]
        restored = pack_unpack(original)
        self.assertEqual(original, restored)

        original = dict(a=Foo_Generic(i=1, j=2, k=3), b=Foo_Special(4, 5), c=[6, 7])
        restored = pack_unpack(original)
        self.assertEqual(original, restored)

    def test_recursiveness(self):
        # Serializable objects can contain other Serializable objects
        obj1 = [1, Foo_Generic(a=2, b=3), Foo_Special(4, 5)]
        obj2 = Foo_Generic(a=obj1, b=Foo_Special(6, 7), c=8)
        obj3 = Foo_Special(obj2, Foo_Generic(a=obj1, b=9, c=obj2, d=10))
        original = [obj1, obj2, obj3]
        restored = pack_unpack(original)
        self.assertEqual(original, restored)

    def test_references(self):
        # Packed references to the same object are unpacked as references to the same object
        obj = Foo_Generic(a=1, b=2, c=3)
        original = [obj, obj]
        restored = pack_unpack(original)
        self.assertIs(restored[0], restored[1])
