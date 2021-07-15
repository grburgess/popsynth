import scipy.stats as stats
import numpy as np
import numba as nb

from popsynth.auxiliary_sampler import AuxiliarySampler, AuxiliaryParameter
from popsynth.distributions.bpl_distribution import bpl


class ParetoAuxSampler(AuxiliarySampler):
    _auxiliary_sampler_name = "ParetoAuxSampler"

    xmin = AuxiliaryParameter(default=1, vmin=0)
    alpha = AuxiliaryParameter(default=1, vmin=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A pareto distribution sampler,
        where property ~ 1 / x^(``alpha`` + 1).

        :param name: Name of the property
        :type name: str
        :param observed: `True` if the property is observed,
            `False` if it is latent. Defaults to `True`
        :type observed: bool
        :param xmin: Minimum value of the pareto
        :type xmin: :class:`AuxiliaryParameter`
        :param alpha: Index of the pareto
        :type alpha: :class:`AuxiliaryParameter`
        :param sigma: Standard deviation of normal distribution
            from which observed values are sampled, if ``observed``
            is `True`
        :type sigma: :class:`AuxiliaryParameter`
        """

        super(ParetoAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size: int):

        self._true_values = (np.random.pareto(self.alpha, size) +
                             1) * self.xmin

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)

        else:

            self._obs_values = self._true_values


class PowerLawAuxSampler(AuxiliarySampler):
    _auxiliary_sampler_name = "PowerLawAuxSampler"

    xmin = AuxiliaryParameter(default=1, vmin=0)
    xmax = AuxiliaryParameter(default=2, vmin=0)
    alpha = AuxiliaryParameter(default=1)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A bounded power law distribution sampler,
        where property ~ x^``alpha``.

        :param name: Name of the property
        :type name: str
        :param observed: `True` if the property is observed,
            `False` if it is latent. Defaults to `True`
        :type observed: bool
        :param xmin: Minimum value of the power law
        :type xmin: :class:`AuxiliaryParameter`
        :param xmax: Maximum value of the power law
        :type xmax: :class:``AuxiliaryParameter
        :param sigma: Standard deviation of normal distribution
            from which observed values are sampled, if ``observed``
            is `True`
        :type sigma: :class:`AuxiliaryParameter`
        """

        super(PowerLawAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size: int):

        self._true_values = _sample_power_law(self.xmin, self.xmax, self.alpha,
                                              size)

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)

        else:

            self._obs_values = self._true_values


class BrokenPowerLawAuxSampler(AuxiliarySampler):
    _auxiliary_sampler_name = "BrokenPowerLawAuxSampler"

    xmin = AuxiliaryParameter(vmin=0)
    alpha = AuxiliaryParameter()
    xbreak = AuxiliaryParameter(vmin=0)
    beta = AuxiliaryParameter()
    xmax = AuxiliaryParameter(vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A broken power law distribution sampler,
        where property ~ x^``alpha`` for x < ``xbreak``,
        and property ~ x^``beta`` for x > ``xbreak``.

        :param name: Name of the property
        :type name: str
        :param observed: `True` if the property is observed,
            `False` if it is latent. Defaults to `True`
        :type observed: bool
        :param xmin: Minimum value of the broken power law
        :type xmin: :class:`AuxiliaryParameter`
        :param xmax: Maximum value of the broken power law
        :type xmax: :class:``AuxiliaryParameter
        :param sigma: Standard deviation of normal distribution
            from which observed values are sampled, if ``observed``
            is `True`
        :type sigma: :class:`AuxiliaryParameter`
        """

        super(BrokenPowerLawAuxSampler, self).__init__(name=name,
                                                       observed=observed)

    def true_sampler(self, size: int):

        u = np.atleast_1d(np.random.uniform(size=size))

        self._true_values = bpl(u, self.xmin, self.xbreak, self.xmax,
                                self.alpha, self.beta)

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)

        else:

            self._obs_values = self._true_values


@nb.njit(fastmath=True)
def _sample_power_law(xmin, xmax, alpha, size):

    out = np.empty(size)

    for i in range(size):

        u = np.random.uniform(0, 1)

        if alpha != -1.0:

            x = np.power(
                (np.power(xmax, alpha + 1) - np.power(xmin, alpha + 1)) * u +
                np.power(xmin, alpha + 1),
                1.0 / (alpha + 1.0),
            )

        else:

            x = xmin * np.exp(u / (1.0 / np.log(xmax / xmin)))

        out[i] = x

    return out
