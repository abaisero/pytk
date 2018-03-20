import functools


class Callback:
    def __init__(self, f):
        self.f = f
        functools.update_wrapper(self, f)
        self.callbacks = []

    def callback(self, g):
        other = Callback(g)
        self.callbacks.append(other)
        return other

    def __call__(self, *args, **kwargs):
        result = self.f(*args, **kwargs)
        for cb in self.callbacks:
            cb(result)
