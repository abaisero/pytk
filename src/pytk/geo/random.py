import numpy as np

from .geo import GeoException, rquat


class Quassian:
    def __init__(self, cov):
        if cov.shape != (4, 4):
            # raise GeoException(f'Cov.shape should be (4, 4);  (is {cov.shape}).')
            raise GeoException('Cov.shape should be (4, 4);  (is {}).'.format(cov.shape))

        self.cov = cov
        self.mean = np.zeros(self.ndim)
        self.rv = multivariate_normal(cov=self.cov)

    @property
    def ndim(self):
        return self.cov.shape[0]

    # TODO what was this?
    # @classmethod
    # def from_ndarray(cls, quats_ndarray, inv_included=False):
    #     # TODO double check cov stuff
    #     if not inv_included:
    #         quats_ndarray =
    #     cov = np.cov(quats_ndarray, rowvar=False)
    #     return Quassian(cov)

    @classmethod
    def from_quat(cls, quats):
        quats_ndarray = np.array([q.wxyz for q in quats])
        return Quassian.from_ndarray(quats_ndarray)

    def sample(self, size=1):
        return rquat(self.rvs(size))
