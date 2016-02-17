import unittest

from pytk.pack import Serializable


class CustomClass_Generic(Serializable):
    def __init__(self, **kwargs):
        self.contents = kwargs

    def __eq__(self, other):
        """ only required for testing """
        return self.contents == other.contents

    def _encode(self):
        """ returns a dictionary with all that is necessary for new object instantiation """
        return self.contents

    @classmethod
    def _decode(cls, data):
        """ reconstructs object instance from data dictionary """
        return cls(**data)


class CustomClass_Special(Serializable):
    def __init__(self, i, j):
        self.i = i
        self.j = j

    def __eq__(self, other):
        return self.i == other.i and self.j == other.j

    def _encode(self):
        """ returns a dictionary with necessary for new object instantiation """
        return dict(i=self.i, j=self.j)

    @classmethod
    def _decode(cls, data):
        """ reconstructs object instance from data dictionary """
        i, j = data['i'], data['j']
        return cls(i, j)


class PackTest(unittest.TestCase):
    def test_builtin_types(self):
        obj = [1, 2, 3]

        packed = Serializable.packb(obj)
        unpacked = Serializable.unpackb(packed)

        self.assertTrue(obj == unpacked)

    def test_custom_class(self):
        obj = CustomClass_Generic(i=1, j=2, k=3)

        packed = Serializable.packb(obj)
        unpacked = Serializable.unpackb(packed)

        self.assertTrue(obj == unpacked)

    def test_mixes_types(self):
        obj = [CustomClass_Generic(i=1, j=2, k=3), CustomClass_Special(2, 3)]

        packed = Serializable.packb(obj)
        unpacked = Serializable.unpackb(packed)

        self.assertTrue(obj == unpacked)

    def test_recursive(self):
        a = [1, CustomClass_Generic(a1=2, a2=3), CustomClass_Special(4, 5)]
        b = CustomClass_Generic(b1=a, b2=CustomClass_Special(6, 7), b3=8)
        c = CustomClass_Special(b, CustomClass_Generic(c1=a, c2=9, c3=b, c4=10))
        obj = [a, b, c]

        packed = Serializable.packb(obj)
        unpacked = Serializable.unpackb(packed)

        self.assertTrue(obj == unpacked)
