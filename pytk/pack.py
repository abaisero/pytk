import msgpack

import pytk.importable as importable


class Serializable(object):
    @property
    def __import_string(self):
        return '{}.{}'.format(self.__module__, self.__class__.__name__)

    @staticmethod
    def _encode(obj):
        if isinstance(obj, Serializable):
            return dict(obj._encode(), __import_string=obj.__import_string)
        return obj

    @staticmethod
    def _decode(data):
        obj = None
        if b'__import_string' in data:
            cls = importable.load_cls(data['__import_string'])
            obj = cls._decode({k: v for k, v in data.iteritems() if k != '__import_string'})
        return data if obj is None else obj

    @staticmethod
    def packb(obj):
        return msgpack.packb(obj, default=Serializable._encode)

    @staticmethod
    def unpackb(stream):
        return msgpack.unpackb(stream, object_hook=Serializable._decode)

    @staticmethod
    def dump(obj, f):
        f.write(Serializable.packb(obj))

    @staticmethod
    def load(f):
        return Serializable.unpackb(f.read())
