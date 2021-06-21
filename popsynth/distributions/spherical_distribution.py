import numpy as np

from popsynth.distribution import SpatialDistribution, DistributionParameter


class SphericalDistribution(SpatialDistribution):
    _distribution_name = "SphericalDistribution"

    def __init__(self, seed=1234, name="sphere", form=None):

        """
        a generic spherical distribution. Can be inherited to
        form more complex spherical distributions

        :param seed: 
        :type seed: 
        :param name: 
        :type name: 
        :param form: 
        :type form: 
        :returns: 

        """
        super(SphericalDistribution, self).__init__(seed=seed,
                                                    name=name,
                                                    form=form)

    def differential_volume(self, r):

        return 4 * np.pi * r * r

    def transform(self, L, r):

        r = r + 1

        return L / (4.0 * np.pi * r * r)


class ConstantSphericalDistribution(SphericalDistribution):
    _distribution_name = "ConstantSphericalDistribution"

    Lambda = DistributionParameter(default=1, vmin=0)

    def __init__(
        self,
        seed=1234,
        name="cons_sphere",
        form=None,
    ):

        """
        A spherical distribution with constant density

        :param seed: 
        :type seed: 
        :param name: 
        :type name: 
        :param form: 
        :type form: 
        :returns: 

        """
        if form is None:
            form = r"\Lambda"

        super(ConstantSphericalDistribution, self).__init__(seed=seed,
                                                            name=name,
                                                            form=form)

    def dNdV(self, distance):

        return self.Lambda


class ZPowerSphericalDistribution(ConstantSphericalDistribution):
    _distribution_name = "ZPowerSphericalDistribution"

    delta = DistributionParameter(default=1)

    def __init__(self, seed=1234, name="zpow_sphere"):

        """
        a spherical population with a power law denisty profile

        :param seed: 
        :type seed: 
        :param name: 
        :type name: 
        :returns: 

        """
        spatial_form = r"\Lambda (z+1)^{\delta}"

        super(ZPowerSphericalDistribution, self).__init__(seed,
                                                          name,
                                                          form=spatial_form)

    def dNdV(self, distance):

        return self.Lambda * np.power(distance + 1.0, self.delta)
