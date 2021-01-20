import numpy as np
from scipy import random as rd

from popsynth.distributions.spherical_distribution import SphericalDistribution
from popsynth.distribution import DistributionParameter


class SpiralGalaxyDistribution(SphericalDistribution):

    rho = DistributionParameter(vmin=0)
    a = DistributionParameter()
    b = DistributionParameter()
    R1 = DistributionParameter()
    R0 = DistributionParameter()

    def __init__(self, seed=1234, name="spiral_galaxy", form=None):

        super(SpiralGalaxyDistribution, self).__init__(seed=seed,
                                                       name=name,
                                                       form=form)

    def dNdV(self, r):

        return (self.rho * np.power(
            (r + self.R1) / (self.R0 + self.R1),
            self.a,
        ) * np.exp(-self.b * (r - self.R0) / (self.R0 + self.R1)))

    def draw_sky_positions(self, size):

        # code thanks to Moritz!

        k = np.array([4.25, 4.25, 4.89, 4.89
                      ])  # Parameter fuer possun [-8.5,0,0], von Wainscoat1992
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

        spiralthetaindec = spiraltheta * 180 / np.pi

        zpos = rd.exponential(height, size=size)

        zpos *= np.random.choice([-1, 1], size=size)

        self._distances = self._distances + np.random.normal(
            0, scale=0.07 * np.abs(self._distances), size=size)

        phi = np.arccos(zpos / np.sqrt(self._distances**2 + zpos**2))

        self._theta = spiraltheta
        self._phi = phi
