import numpy as np
import numba as nb
import math

from popsynth.distribution import SpatialDistribution, DistributionParameter
from popsynth.utils.cosmology import cosmology


class CosmologicalDistribution(SpatialDistribution):
    def __init__(self,
                 seed=1234,
                 name="cosmo",
                 form=None,
                 truth={},
                 is_rate=True):

        super(CosmologicalDistribution, self).__init__(
            seed=seed,
            name=name,
            form=form,
        )
        self._is_rate = is_rate

    def differential_volume(self, z):

        return cosmology.differential_comoving_volume(z)

    def transform(self, L, z):

        return L / (4.0 * np.pi * cosmology.luminosity_distance(z)**2)

    def time_adjustment(self, z):
        if self._is_rate:
            return 1 + z
        else:
            return 1.0


class SFRDistribution(CosmologicalDistribution):

    r0 = DistributionParameter(vmin=0)
    rise = DistributionParameter()
    decay = DistributionParameter()
    peak = DistributionParameter(vmin=0)

    def __init__(self, seed=1234, name="sfr", is_rate=True):

        spatial_form = r"\rho_0 \frac{1+r \cdot z}{1+ \left(z/p\right)^d}"

        super(SFRDistribution, self).__init__(
            seed=seed,
            name=name,
            form=spatial_form,
            is_rate=is_rate,
        )

    def dNdV(self, z):
        return _dndv(
            z,
            self.r0,
            self.rise,
            self.decay,
            self.peak,
        )


@nb.njit(fastmath=True)
def _dndv(z, r0, rise, decay, peak):
    top = 1.0 + rise * z
    bottom = 1.0 + np.power(z / peak, decay)

    return r0 * top / bottom


class ZPowerCosmoDistribution(CosmologicalDistribution):

    Lambda = DistributionParameter(default=1, vmin=0)
    delta = DistributionParameter()

    def __init__(
        self,
        seed=1234,
        name="zpow_cosmo",
        is_rate=True,
    ):

        spatial_form = r"\Lambda (z+1)^{\delta}"

        super(ZPowerCosmoDistribution, self).__init__(
            seed=seed,
            name=name,
            form=spatial_form,
            is_rate=is_rate,
        )

    def dNdV(self, distance):

        return self.Lambda * np.power(distance + 1.0, self.delta)


# class MergerDistribution(CosmologicalDistribution):
#     def __init__(self, r0, td, sigma, r_max=10, seed=1234, name="merger"):

#         spatial_form = r"\rho_0 \frac{1+r \cdot z}{1+ \left(z/p\right)^d}"

#         super(MergerDistribution, self).__init__(
#             r_max=r_max, seed=seed, name=name, form=spatial_form
#         )

#         self._td = td
#         self._sigma = sigma

#     def _sfr(self, z):

#         top = 0.01 + 0.12 * z
#         bottom = 1.0 + np.power(z / 3.23, 4.66)
#         return top / bottom

#     def _delay_time(self, tau):
#         return np.exp(
#             -((np.log(tau) - np.log(self._td)) ** 2) / (2 * self._sigma ** 2)
#         ) / (np.sqrt(2 * np.pi) * self._sigma)

#     def dNdV(self, z):

#         integrand = lambda x: self._sfr(x) * self._delay_time(
#             cosmo.lookback_time(x).value
#         )
#         out = []

#         try:
#             return self.r0 * integrate.quad(integrand, z, np.infty)[0]

#         except:
#             for zz in z:

#                 out.append(integrate.quad(integrand, zz, np.infty)[0])

#             return self.r0 * np.array(out)
