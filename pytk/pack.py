import msgpack
import msgpack_numpy as msgpack_np
# The patch is overwritten anyway
# msgpack_np.patch()

import pytk.importable as importable


class Serializable(object):
    """ Inherit from this class if you want your custom objects to be serializable """

    def _encode(self):
        """ returns a dictionary with all that is necessary for new object instantiation """
        raise NotImplementedError

    @classmethod
    def _decode(cls, data):
        """ reconstructs object instance from data dictionary """
        raise NotImplementedError


def _import_string(obj):
    return '{}.{}'.format(obj.__module__, obj.__class__.__name__)


def _encode_default(obj):
    if isinstance(obj, Serializable):
        return dict(_code=obj._encode(), _id=id(obj), _import_string=_import_string(obj))
    return msgpack_np.encode(obj)
    # return obj

_obj_cache = dict()


def _decode_object_hook(data):
    obj = None
    keys = ('_code', '_import_string', '_id')
    if all(k in data for k in keys):
        _code, _import_string, _id = (data[k] for k in keys)
        if _id in _obj_cache:
            obj = _obj_cache[_id]
        else:
            cls = importable.load_cls(_import_string)
            obj = cls._decode(_code)
            _obj_cache[_id] = obj
    if obj is None:
        obj = msgpack_np.decode(data)
    return obj


def packb(obj):
    """ encodes object and returns byte stream """
    return msgpack.packb(obj, default=_encode_default)


def unpackb(stream):
    """ decodes byte stream and returns object """
    global _obj_cache
    _obj_cache = dict()
    return msgpack.unpackb(stream, object_hook=_decode_object_hook)


def dump(obj, f):
    """ dumps object into file """
    f.write(packb(obj))


def load(f):
    """ loads object from file """
    return unpackb(f.read())

from pathlib import Path


class Parcel(object):

    def __init__(self):
        self.labels = []

    @property
    def label(self):
        return '.'.join(['Parcel'] + self.labels + ['pack'])

    def clear(self):
        del self.labels[:]

    def seal(self, obj, path=None):
        if path is None:
            path = Path.cwd()
        with open(str(path / self.label), 'wb') as f:
            dump(obj, f)

    def unseal(self, path=None):
        if path is None:
            path = Path.cwd()
        with open(str(path / self.label), 'rb') as f:
            return load(f)

    def __str__(self):
        return self.label
