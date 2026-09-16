import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats

from popsynth.distribution import DistributionParameter, LuminosityDistribution
from popsynth.distributions.cosmological_distribution import (
    CosmologicalDistribution, )


class LogUniLuminiosityDistribution(LuminosityDistribution):
    _distribution_name = "LogUniLuminiosityDistribution"

    Lmin = DistributionParameter(vmin=0)

    Lmax = DistributionParameter(vmin=0)

    def __init__(self, seed: int = 1234, name: str = "logunilum"):
        """A broken power law luminosity distribution.

        L ~ L^``alpha`` for L <= ``Lbreak``
        L ~ L^``beta`` for L > ``Lbreak``

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param Lmin: Minimum value of the luminosity
        :type Lmin: :class:`DistributionParameter`
        :param Lmax: Maximum value of the luminosity
        :type Lmax: :class:`DistributionParameter`
        """
        lf_form = r"\begin{cases} C L^{\alpha} & \mbox{if } L"
        lf_form += r"\leq L_\mathrm{break},\\ C L^{\beta} "
        lf_form += r"L_\mathrm{break}^{\alpha - \beta}"
        lf_form += r" & \mbox{if } L > L_\mathrm{break}. \end{cases}"

        super(LogUniLuminiosityDistribution, self).__init__(seed=seed,
                                                            name=name,
                                                            form=lf_form)

    def phi(self, L):

        return 1.0 / (L * np.log(self.Lmax / self.Lmin))

    def draw_luminosity(self, size=1):

        return stats.loguniform.rvs(self.Lmin, self.Lmax, size=size)


class UniformCosmoDistribution(CosmologicalDistribution):
    _distribution_name = "UniformCosmoDistribution"

    r0 = DistributionParameter(vmin=0, is_normalization=True)
    zmin = DistributionParameter(vmin=0)
    zmax = DistributionParameter(vmin=0)

    def __init__(
        self,
        seed: int = 1234,
        name: str = "uniform_cosmo",
        is_rate: bool = True,
    ):
        """
        A cosmological distribution where the density
        evolves uniformly.

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

        super(UniformCosmoDistribution, self).__init__(
            seed=seed,
            name=name,
            form=spatial_form,
            is_rate=is_rate,
        )

    def dNdV(self, distance):

        return self.r0 / (self.zmax - self.zmin)
