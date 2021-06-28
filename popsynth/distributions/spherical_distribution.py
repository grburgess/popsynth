import numpy as np

from popsynth.distribution import SpatialDistribution, DistributionParameter


class SphericalDistribution(SpatialDistribution):
    _distribution_name = "SphericalDistribution"

    def __init__(
        self,
        seed: int = 1234,
        name: str = "sphere",
        form: str = None,
    ):
        """
        A generic spherical distribution. Can be inherited to
        form more complex spherical distributions

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param form: Mathematical description of distribution
        :type form: str
        """
        super(SphericalDistribution, self).__init__(
            seed=seed,
            name=name,
            form=form,
        )

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
        seed: int = 1234,
        name: str = "cons_sphere",
        form: str = None,
    ):
        """
        A spherical distribution with constant density.

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param form: Mathematical description of distribution
        :type form: str
        :param Lambda: Density per unit volume
        :type Lambda: :class:`DistributionParameter`
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

    def __init__(self, seed: int = 1234, name: str = "zpow_sphere"):
        """
        A spherical distribution with a power law density profile.

        ``Lambda`` (1+r)^``delta``

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param form: Mathematical description of distribution
        :type form: str
        :param delta: Index of power law distribution
        :type delta: :class:`DistributionParameter`
        """
        spatial_form = r"\Lambda (1+r)^{\delta}"

        super(ZPowerSphericalDistribution, self).__init__(
            seed,
            name,
            form=spatial_form,
        )

    def dNdV(self, distance):

        return self.Lambda * np.power(distance + 1.0, self.delta)
