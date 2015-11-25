import numpy as np

def index_except(slicelen, ex):
    return ~(np.arange(slicelen) == ex)


