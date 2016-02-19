import unittest

import pytk.pack as pack


class CustomClass_Generic(pack.Serializable):
    def __init__(self, **kwargs):
        self.contents = kwargs

    def __str__(self):
        """ Only for debugging the test """
        contents_str = ['{}={}'.format(k, v) for k, v in self.contents.iteritems()]
        return '(CustomClass_Generic ' + ', '.join(contents_str) + ')'

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


class CustomClass_Special(pack.Serializable):
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def __str__(self):
        return '(CustomClass_Special i={}, j={})'.format(self.i, self.j)

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


class PackTest(unittest.TestCase):
    def pack_unpack(self, obj):
        return pack.unpackb(pack.packb(obj))

    def test_builtin_types(self):
        """ Builtin types can be unpacked """
        original = 1
        restored = self.pack_unpack(original)
        self.assertEqual(original, restored)

        original = [1, 2, 3]
        restored = self.pack_unpack(original)
        self.assertEqual(original, restored)

        original = dict(a=1, b=2)
        restored = self.pack_unpack(original)
        self.assertEqual(original, restored)

    def test_custom_types(self):
        """ Serializable objects can be unpacked """
        original = CustomClass_Generic(i=1, j=2, k=3)
        restored = self.pack_unpack(original)
        self.assertEqual(original, restored)

        original = CustomClass_Special(1, 2)
        restored = self.pack_unpack(original)
        self.assertEqual(original, restored)

    def test_mixed_types(self):
        """ Objects can contain other objects """
        original = [CustomClass_Generic(i=1, j=2, k=3), CustomClass_Special(4, 5), [6, 7]]
        restored = self.pack_unpack(original)
        self.assertEqual(original, restored)

        original = dict(a=CustomClass_Generic(i=1, j=2, k=3), b=CustomClass_Special(4, 5), c=[6, 7])
        restored = self.pack_unpack(original)
        self.assertEqual(original, restored)

    def test_recursiveness(self):
        """ Serializable objects can contain other Serializable objects """
        obj1 = [1, CustomClass_Generic(a=2, b=3), CustomClass_Special(4, 5)]
        obj2 = CustomClass_Generic(a=obj1, b=CustomClass_Special(6, 7), c=8)
        obj3 = CustomClass_Special(obj2, CustomClass_Generic(a=obj1, b=9, c=obj2, d=10))
        original = [obj1, obj2, obj3]
        restored = self.pack_unpack(original)
        self.assertEqual(original, restored)

    def test_references(self):
        """ Packed references to the same object are unpacked as references to the same object """
        obj = CustomClass_Generic(a=1, b=2, c=3)
        original = [obj, obj]
        restored = self.pack_unpack(original)
        self.assertIs(restored[0], restored[1])
