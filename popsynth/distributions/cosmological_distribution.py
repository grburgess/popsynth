import numpy as np
import math
from astropy.constants import c as sol
from numba import jit, njit
from astropy.cosmology import WMAP9 as cosmo

from popsynth.distribution import SpatialDistribution, DistributionParameter


sol = sol.value

h0 = 67.7
dh = sol * 1.0e-3 / h0
Om = 0.307
Om_reduced = (1 - Om) / Om
Om_sqrt = np.sqrt(Om)
Ode = 1 - Om - (cosmo.Onu0 + cosmo.Ogamma0)


@njit(fastmath=True)
def Phi(x):
    x2 = x * x
    x3 = x * x * x
    top = 1.0 + 1.320 * x + 0.441 * x2 + 0.02656 * x3
    bottom = 1.0 + 1.392 * x + 0.5121 * x2 + 0.03944 * x3
    return top / bottom


@njit(fastmath=True)
def xx(z):
    return Om_reduced / np.power(1.0 + z, 3)


@njit(fastmath=True)
def luminosity_distance(z):
    x = xx(z)
    z1 = 1.0 + z
    val = (
        (2 * dh * z1 / Om_sqrt) * (Phi(xx(0)) - 1.0 / (np.sqrt(z1)) * Phi(x)) * 3.086e24
    )  # in cm
    return val


@njit(fastmath=True)
def a(z):
    return np.sqrt(np.power(1 + z, 3.0) * Om + Ode)


@njit(fastmath=True)
def comoving_transverse_distance(z):
    return luminosity_distance(z) / (1.0 + z)


@njit(fastmath=True)
def differential_comoving_volume(z):
    td = comoving_transverse_distance(z) / 3.086e24
    return (dh * td * td / a(z)) * 1e-9  # Gpc^3


class CosmologicalDistribution(SpatialDistribution):
    def __init__(self, seed=1234, name="cosmo", form=None, truth={}, is_rate=True):

        super(CosmologicalDistribution, self).__init__(
            seed=seed, name=name, form=form,
        )
        self._is_rate = is_rate

    def differential_volume(self, z):

        td = comoving_transverse_distance(z) / 3.086e24
        return (dh * td * td / a(z)) * 1e-9  # Gpc^3

    def transform(self, L, z):

        return L / (4.0 * np.pi * luminosity_distance(z) ** 2)

    def time_adjustment(self, z):
        if self._is_rate:
            return 1 + z
        else:
            return 1.0


class SFRDistribtution(CosmologicalDistribution):

    r0 = DistributionParameter(vmin=0)
    rise = DistributionParameter()
    decay = DistributionParameter()
    peak = DistributionParameter(vmin=0)

    def __init__(self, seed=1234, name="sfr", is_rate=True):

        spatial_form = r"\rho_0 \frac{1+r \cdot z}{1+ \left(z/p\right)^d}"

        super(SFRDistribtution, self).__init__(
            seed=seed, name=name, form=spatial_form, is_rate=is_rate,
        )

    def dNdV(self, z):
        return _dndv(z, self.r0, self.rise, self.decay, self.peak,)


@njit(fastmath=True)
def _dndv(z, r0, rise, decay, peak):
    top = 1.0 + rise * z
    bottom = 1.0 + np.power(z / peak, decay)

    return r0 * top / bottom


class ZPowerCosmoDistribution(CosmologicalDistribution):

    Lambda = DistributionParameter(default=1, vmin=0)
    delta = DistributionParameter()

    def __init__(
        self, seed=1234, name="zpow_cosmo", is_rate=True,
    ):

        spatial_form = r"\Lambda (z+1)^{\delta}"

        super(ZPowerCosmoDistribution, self).__init__(
            seed=seed, name=name, form=spatial_form, is_rate=is_rate,
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
