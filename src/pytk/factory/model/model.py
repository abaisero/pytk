

# TODO inherits from distribution?

class Model:
    def __init__(self, *yfactories, cond=None):
        if cond is None:
            cond = ()
        xfactories = cond
        xyfactories = xfactories + yfactories

        self.xfactories = xfactories
        self.yfactories = yfactories
        self.xyfactories = xyfactories

        self.nx = len(xfactories)
        self.ny = len(yfactories)
        self.nxy = len(xyfactories)
