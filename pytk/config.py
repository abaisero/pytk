import importlib


cmap = dict()


def load(module_name):
    global cmap
    module = importlib.import_module(module_name)
    cmap.update(module.cmap)
        # (attr, getattr(module, attr))
        #     for attr in dir(module)
        #         if not attr.startswith('__')
    # )
    return cmap
