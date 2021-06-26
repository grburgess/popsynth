import numpy as np
from scipy import random as rd

from popsynth.distribution import DistributionParameter
from popsynth.distributions.spherical_distribution import SphericalDistribution


class SpiralGalaxyDistribution(SphericalDistribution):
    _distribution_name = "SpiralGalaxyDistribution"

    rho = DistributionParameter(vmin=0)
    a = DistributionParameter()
    b = DistributionParameter()
    R1 = DistributionParameter()
    R0 = DistributionParameter()

    def __init__(
        self,
        seed: int = 1234,
        name: str = "spiral_galaxy",
        form: str = None,
    ):
        """
        A spiral galaxy spatial distribution.

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param form: Mathematical description of distribution
        :type form: str
        :param rho: Local density
        :type rho: :class:`DistributionParameter`
        :param a: Shape parameter
        :type a: :class:`DistributionParameter`
        :param b: Shape parameter
        :type b: :class:`DistributionParameter`
        :param R1: Scale parameter
        :type R1: :class:`DistributionParameter`
        :param R0: Scale parameter
        :type R0: :class:`DistributionParameter`
        """

        spatial_form = r"\rho \Big(\frac{r + R_1}{R_0+R_1}\Big)^a "
        spatial_form += r"\exp\Big[-b \frac{r-R_0}{R_0 + R_1}\Big]"

        super(SpiralGalaxyDistribution, self).__init__(
            seed=seed,
            name=name,
            form=form,
        )

    def dNdV(self, r):

        return (self.rho * np.power(
            (r + self.R1) / (self.R0 + self.R1),
            self.a,
        ) * np.exp(-self.b * (r - self.R0) / (self.R0 + self.R1)))

    def draw_sky_positions(self, size):
        """
        Based on Wainscoat 1992 and
        Faucher-Giguere 2007.

        Code thanks to Mortiz Pleintinger.
        """

        # Parameters for possun [-8.5,0,0], from Wainscoat 1992
        k = np.array([4.25, 4.25, 4.89, 4.89])
        r0 = np.array([3.48, 3.48, 4.90, 4.90])
        theta0 = np.array([0.0, 3.141, 2.525, 5.666])  # Wainscoat 1992
        height = 0.3  # kpc

        idx = np.random.randint(4, size=size)

        ks = k[idx]
        r0s = r0[idx]
        theta0s = theta0[idx]

        tet = ks * np.log(self._distances / r0s) + theta0s

        winklcor = np.random.uniform(0.0, 2 * np.pi, size=size)

        corrtet = winklcor * np.exp(
            -0.35 * self._distances)  # Faucher-Giguere 2007

        spiraltheta = tet + corrtet  # Faucher-Giguere 2007

        zpos = rd.exponential(height, size=size)

        zpos *= np.random.choice([-1, 1], size=size)

        self._distances = self._distances + np.random.normal(
            0, scale=0.07 * np.abs(self._distances), size=size)

        phi = np.arccos(zpos / np.sqrt(self._distances**2 + zpos**2))

        self._theta = spiraltheta

        self._phi = phi
