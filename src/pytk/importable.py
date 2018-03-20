import importlib
import re


__import_string_regex = re.compile('(?P<module_name>.+)\.(?P<class_name>[^\.]+)')


def load_cls(import_string):
    match = __import_string_regex.match(import_string)
    module_name, class_name = match.group('module_name', 'class_name')

    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
    except Exception as e:
        e.args = e.args + (
            'Error while instantiating class \'{}\' from module \'{}\''
            .format(class_name, module_name),)
        raise
    return cls


def load_obj(import_string, *args, **kwargs):
    return load_cls(import_string)(*args, **kwargs)
