import numpy as np
import math

from astropy.cosmology import WMAP9 as cosmo

from popsynth.distribution import SpatialDistribution

# from popsynth.utils.package_data import copy_package_data

import scipy.integrate as integrate
from astropy.constants import c as sol

from numba import jit, njit


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
    def __init__(
        self, r_max=10, seed=1234, name="cosmo", form=None, truth={}, is_rate=True
    ):

        super(CosmologicalDistribution, self).__init__(
            r_max=r_max, seed=seed, name=name, form=form, truth=truth
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
    def __init__(
        self, r0, rise, decay, peak, r_max=10, seed=1234, name="sfr", is_rate=True
    ):

        spatial_form = r"\rho_0 \frac{1+r \cdot z}{1+ \left(z/p\right)^d}"

        truth = dict(r0=r0, rise=rise, decay=decay, peak=peak)

        super(SFRDistribtution, self).__init__(
            r_max=r_max,
            seed=seed,
            name=name,
            form=spatial_form,
            truth=truth,
            is_rate=is_rate,
        )

        self._construct_distribution_params(r0=r0, rise=rise, decay=decay, peak=peak)

    def dNdV(self, z):
        return _dndv(
            z,
            self._params["r0"],
            self._params["rise"],
            self._params["decay"],
            self._params["peak"],
        )

    def __get_r0(self):
        """Calculates the 'r0' property."""
        return self._params["r0"]

    def ___get_r0(self):
        """Indirect accessor for 'r0' property."""
        return self.__get_r0()

    def __set_r0(self, r0):
        """Sets the 'r0' property."""
        self.set_distribution_params(
            r0=r0, rise=self.rise, decay=self.decay, peak=self.peak
        )

    def ___set_r0(self, r0):
        """Indirect setter for 'r0' property."""
        self.__set_r0(r0)

    r0 = property(___get_r0, ___set_r0, doc="""Gets or sets the r0.""")

    def __get_rise(self):
        """Calculates the 'rise' property."""
        return self._params["rise"]

    def ___get_rise(self):
        """Indirect accessor for 'rise' property."""
        return self.__get_rise()

    def __set_rise(self, rise):
        """Sets the 'rise' property."""
        self.set_distribution_params(
            r0=self.r0, rise=rise, decay=self.decay, peak=self.peak
        )

    def ___set_rise(self, rise):
        """Indirect setter for 'rise' property."""
        self.__set_rise(rise)

    rise = property(___get_rise, ___set_rise, doc="""Gets or sets the rise.""")

    def __get_decay(self):
        """Calculates the 'decay' property."""
        return self._params["decay"]

    def ___get_decay(self):
        """Indirect accessor for 'decay' property."""
        return self.__get_decay()

    def __set_decay(self, decay):
        """Sets the 'decay' property."""
        self.set_distribution_params(
            r0=self.r0, rise=self.rise, decay=decay, peak=self.peak
        )

    def ___set_decay(self, decay):
        """Indirect setter for 'decay' property."""
        self.__set_decay(decay)

    decay = property(___get_decay, ___set_decay, doc="""Gets or sets the decay.""")

    def __get_peak(self):
        """Calculates the 'peak' property."""
        return self._params["peak"]

    def ___get_peak(self):
        """Indirect accessor for 'peak' property."""
        return self.__get_peak()

    def __set_peak(self, peak):
        """Sets the 'peak' property."""
        self.set_distribution_params(
            r0=self.r0, rise=self.rise, decay=self.decay, peak=peak
        )

    def ___set_peak(self, peak):
        """Indirect setter for 'peak' property."""
        self.__set_peak(peak)

    peak = property(___get_peak, ___set_peak, doc="""Gets or sets the peak.""")


@njit(fastmath=True)
def _dndv(z, r0, rise, decay, peak):
    top = 1.0 + rise * z
    bottom = 1.0 + np.power(z / peak, decay)

    return r0 * top / bottom


class ZPowerCosmoDistribution(CosmologicalDistribution):
    def __init__(
        self,
        Lambda=1.0,
        delta=1.0,
        r_max=10.0,
        seed=1234,
        name="zpow_cosmo",
        is_rate=True,
    ):

        spatial_form = r"\Lambda (z+1)^{\delta}"

        truth = dict(Lambda=Lambda, delta=delta)

        super(ZPowerCosmoDistribution, self).__init__(
            r_max=r_max,
            seed=seed,
            name=name,
            form=spatial_form,
            truth=truth,
            is_rate=is_rate,
        )

        self._construct_distribution_params(Lambda=Lambda, delta=delta)

    def __get_delta(self):
        """Calculates the 'delta' property."""
        return self._params["delta"]

    def ___get_delta(self):
        """Indirect accessor for 'delta' property."""
        return self.__get_delta()

    def __set_delta(self, delta):
        """Sets the 'delta' property."""
        self.set_distribution_params(delta=delta)

    def ___set_delta(self, delta):
        """Indirect setter for 'delta' property."""
        self.__set_delta(delta)

    delta = property(___get_delta, ___set_delta, doc="""Gets or sets the delta.""")

    def dNdV(self, distance):

        return self._params["Lambda"] * np.power(distance + 1.0, self._params["delta"])


class MergerDistribution(CosmologicalDistribution):
    def __init__(self, r0, td, sigma, r_max=10, seed=1234, name="merger"):

        spatial_form = r"\rho_0 \frac{1+r \cdot z}{1+ \left(z/p\right)^d}"

        super(MergerDistribution, self).__init__(
            r_max=r_max, seed=seed, name=name, form=spatial_form
        )

        self._construct_distribution_params(r0=r0, td=td, sigma=sigma)

        self._td = td
        self._sigma = sigma

    def _sfr(self, z):

        top = 0.01 + 0.12 * z
        bottom = 1.0 + np.power(z / 3.23, 4.66)
        return top / bottom

    def _delay_time(self, tau):
        return np.exp(
            -((np.log(tau) - np.log(self._td)) ** 2) / (2 * self._sigma ** 2)
        ) / (np.sqrt(2 * np.pi) * self._sigma)

    def dNdV(self, z):

        integrand = lambda x: self._sfr(x) * self._delay_time(
            cosmo.lookback_time(x).value
        )
        out = []

        try:
            return self.r0 * integrate.quad(integrand, z, np.infty)[0]

        except:
            for zz in z:

                out.append(integrate.quad(integrand, zz, np.infty)[0])

            return self.r0 * np.array(out)

    def __get_r0(self):
        """Calculates the 'r0' property."""
        return self._params["r0"]

    def ___get_r0(self):
        """Indirect accessor for 'r0' property."""
        return self.__get_r0()

    def __set_r0(self, r0):
        """Sets the 'r0' property."""
        self.set_distribution_params(
            r0=r0, rise=self.rise, decay=self.decay, peak=self.peak
        )

    def ___set_r0(self, r0):
        """Indirect setter for 'r0' property."""
        self.__set_r0(r0)

    r0 = property(___get_r0, ___set_r0, doc="""Gets or sets the r0.""")

    # def __get_rise(self):
    #          """Calculates the 'rise' property."""
    #          return self._spatial_params['rise']

    # def ___get_rise(self):
    #      """Indirect accessor for 'rise' property."""
    #      return self.__get_rise()

    # def __set_rise(self, rise):
    #      """Sets the 'rise' property."""
    #      self.set_spatial_distribution_params(r0=self.r0, rise=rise, decay=self.decay, peak=self.peak)

    # def ___set_rise(self, rise):
    #      """Indirect setter for 'rise' property."""
    #      self.__set_rise(rise)

    # rise = property(___get_rise, ___set_rise,
    #                  doc="""Gets or sets the rise.""")

    # def __get_decay(self):
    #          """Calculates the 'decay' property."""
    #          return self._spatial_params['decay']

    # def ___get_decay(self):
    #      """Indirect accessor for 'decay' property."""
    #      return self.__get_decay()

    # def __set_decay(self, decay):
    #      """Sets the 'decay' property."""
    #      self.set_spatial_distribution_params(r0=self.r0, rise=self.rise, decay=decay, peak=self.peak)

    # def ___set_decay(self, decay):
    #      """Indirect setter for 'decay' property."""
    #      self.__set_decay(decay)

    # decay = property(___get_decay, ___set_decay,
    #                  doc="""Gets or sets the decay.""")

    # def __get_peak(self):
    #     """Calculates the 'peak' property."""
    #     return self._spatial_params['peak']

    # def ___get_peak(self):
    #      """Indirect accessor for 'peak' property."""
    #      return self.__get_peak()

    # def __set_peak(self, peak):
    #      """Sets the 'peak' property."""
    #      self.set_spatial_distribution_params(r0=self.r0, rise=self.rise, decay=self.decay, peak=peak)

    # def ___set_peak(self, peak):
    #      """Indirect setter for 'peak' property."""
    #      self.__set_peak(peak)

    # peak = property(___get_peak, ___set_peak,
    #                  doc="""Gets or sets the peak.""")
