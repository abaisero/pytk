import more_itertools as mitt

class PowerDist(object):
    def __init__(self, X):
        self.X = X
        self.PX = list(mitt.powerset(X))

    def __call__(self, x):

