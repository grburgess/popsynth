import numpy as np
import numba as nb
from typing import Dict, Any
from popsynth.distribution import SpatialDistribution, DistributionParameter
from popsynth.utils.cosmology import cosmology


class CosmologicalDistribution(SpatialDistribution):
    _distribution_name = "CosmologicalDistribution"

    def __init__(
        self,
        seed: int = 1234,
        name: str = "cosmo",
        form: str = None,
        truth: Dict[str, Any] = {},
        is_rate: bool = True,
    ):
        """
        Base class for cosmological spatial distributions.

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param form: Mathematical description of distribution
        :type form: str
        :param truth: True values of parameters
        :type truth: dict[str, Any]
        :param is_rate: `True` if modelling a population of transient events,
            `False` if modelling a population of steady-state objects.
            Affects the ``time_adjustment`` method used in cosmo calculations.
            Default is `True`.
        :type is_rate: bool
        """
        super(CosmologicalDistribution, self).__init__(
            seed=seed,
            name=name,
            form=form,
        )
        self._is_rate = is_rate

    def differential_volume(self, z):
        """
        Differential comoving volume in Gpc^3 sr^-1.

        dV/dzdOmega

        :param z: Redshift
        :returns: The differential comoving volume in
            Gpc^-3 sr^-1.
        """

        return cosmology.differential_comoving_volume(z)

    def transform(self, L, z):
        """
        Transformation from luminosity to energy flux.

        L / 4 pi dL^2

        dL is in cm. Therefore for L in erg s^-1 returns
        flux in erg cm^-2 s^-1.

        :param L: Luminosity
        :param z: Redshift
        :returns: Flux
        """

        return L / (4.0 * np.pi * cosmology.luminosity_distance(z)**2)

    def time_adjustment(self, z):
        """
        Time adjustment factor to handle both
        transient and steady-state populations.

        :param z: Redshift
        :returns: Appropriate factor depending on ``is_rate``
        """
        if self._is_rate:

            return 1 + z

        else:

            return 1.0


class SFRDistribution(CosmologicalDistribution):
    _distribution_name = "SFRDistribution"

    r0 = DistributionParameter(vmin=0)
    a = DistributionParameter(vmin=0)
    rise = DistributionParameter()
    decay = DistributionParameter()
    peak = DistributionParameter(vmin=0)

    def __init__(self,
                 seed: int = 1234,
                 name: str = "sfr",
                 is_rate: bool = True):
        """
        A star-formation like distribution of the form
        presented in Cole et al. 2001.

        ``r0``(``a``+``rise``z)/(1 + (z/``peak``)^``decay``)

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param is_rate: `True` if modelling a population of transient events,
            `False` if modelling a population of steady-state objects.
            Affects the ``time_adjustment`` method used in cosmo calculations.
            Default is `True`.
        :type is_rate: bool
        :param r0: The local density in units of Gpc^-3
        :type r0: :class:`DistributionParameter`
        :param a: Offset at z=0
        :type a: :class:`DistributionParameter`
        :param rise: Rise at low z
        :type rise: :class:`DistributionParameter`
        :param decay: Decay at high z
        :type decay: :class:`DistributionParameter`
        :param peak: Peak of z distribution
        :type peak: :class:`DistributionParameter`
        """
        spatial_form = r"\rho_0 \frac{a+r \cdot z}{1+ \left(z/p\right)^d}"

        super(SFRDistribution, self).__init__(
            seed=seed,
            name=name,
            form=spatial_form,
            is_rate=is_rate,
        )

    def dNdV(self, z):
        return _sfr_dndv(
            z,
            self.r0,
            self.a,
            self.rise,
            self.decay,
            self.peak,
        )


@nb.njit(fastmath=True)
def _sfr_dndv(z, r0, a, rise, decay, peak):
    top = a + rise * z
    bottom = 1.0 + np.power(z / peak, decay)

    return r0 * top / bottom


class ZPowerCosmoDistribution(CosmologicalDistribution):
    _distribution_name = "ZPowerCosmoDistribution"

    Lambda = DistributionParameter(default=1, vmin=0)
    delta = DistributionParameter()

    def __init__(
        self,
        seed: int = 1234,
        name: str = "zpow_cosmo",
        is_rate: bool = True,
    ):
        """
        A cosmological distribution where the density
        evolves as a power law.

        ``Lambda`` (1+z)^``delta``

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param is_rate: `True` if modelling a population of transient events,
            `False` if modelling a population of steady-state objects.
            Affects the ``time_adjustment`` method used in cosmo calculations.
            Default is `True`.
        :type is_rate: bool
        :param Lambda: The local density in units of Gpc^-3
        :type Lambda: :class:`DistributionParameter`
        :param delta: The index of the power law
        :type delta: :class:`DistributionParameter`
        """
        spatial_form = r"\Lambda (z+1)^{\delta}"

        super(ZPowerCosmoDistribution, self).__init__(
            seed=seed,
            name=name,
            form=spatial_form,
            is_rate=is_rate,
        )

    def dNdV(self, distance):

        return _zp_dndv(distance, self.Lambda, self.delta)


@nb.njit(fastmath=True)
def _zp_dndv(z, Lambda, delta):
    return Lambda * np.power(z + 1, delta)


# class MergerDistribution(CosmologicalDistribution):
# _distribution_name = "MergerDistribution"
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
