import numpy as np

from popsynth.distribution import SpatialDistribution, DistributionParameter


class FlatlandDistribution(SpatialDistribution):

    Lambda = DistributionParameter(default=1, vmin=0)

    def __init__(self, seed: int = 1234, name: str = "flatland", form=None):

        super(FlatlandDistribution, self).__init__(seed=seed,
                                                   name=name,
                                                   form=form)

    def differential_volume(self, r):

        return 1

    def transform(self, L, r):

        return L

    def dNdV(self, distance):

        return self.Lambda
