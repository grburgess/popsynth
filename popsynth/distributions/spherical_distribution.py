import numpy as np

from popsynth.distribution import SpatialDistribution, DistributionParameter


class SphericalDistribution(SpatialDistribution):
    def __init__(self, seed=1234, name="sphere", form=None):

        super(SphericalDistribution, self).__init__(seed=seed,
                                                    name=name,
                                                    form=form)

    def differential_volume(self, r):

        return 4 * np.pi * r * r

    def transform(self, L, r):

        r = r + 1

        return L / (4.0 * np.pi * r * r)


class ConstantSphericalDistribution(SphericalDistribution):

    Lambda = DistributionParameter(default=1, vmin=0)

    def __init__(
        self,
        seed=1234,
        name="cons_sphere",
        form=None,
    ):

        if form is None:
            form = r"\Lambda"

        super(ConstantSphericalDistribution, self).__init__(seed=seed,
                                                            name=name,
                                                            form=form)

    def dNdV(self, distance):

        return self.Lambda


class ZPowerSphericalDistribution(ConstantSphericalDistribution):

    delta = DistributionParameter(default=1)

    def __init__(self, seed=1234, name="zpow_sphere"):

        spatial_form = r"\Lambda (z+1)^{\delta}"

        super(ZPowerSphericalDistribution, self).__init__(seed,
                                                          name,
                                                          form=spatial_form)

    def dNdV(self, distance):

        return self.Lambda * np.power(distance + 1.0, self.delta)
