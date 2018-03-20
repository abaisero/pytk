from abc import ABCMeta, abstractmethod

import inspect
import types


# TODO this is not nice....  what does this represent..?
class Distribution(metaclass=ABCMeta):
    def __init__(self, *, cond=None):
        if cond is None:
            cond = ()
        xfactories = cond

        self.xfactories = xfactories
        self.nx = len(xfactories)
