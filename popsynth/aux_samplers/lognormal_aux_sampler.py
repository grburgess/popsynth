import numpy as np
import scipy.stats as stats

from popsynth.auxiliary_sampler import AuxiliaryParameter, AuxiliarySampler


class LogNormalAuxSampler(AuxiliarySampler):
    _auxiliary_sampler_name = "LogNormalAuxSampler"

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A Log normal sampler,
        where property ~ e^N(``mu``, ``sigma``).

        :param name: Name of the property
        :type name: str
        :param observed: `True` if the property is observed,
            `False` if it is latent. Defaults to `True`
        :type observed: bool
        :param mu: Mean of the lognormal
        :type mu: :class:`AuxiliaryParameter`
        :param tau: Standard deviation of the lognormal
        :type tau: :class:`AuxiliaryParameter`
        :param sigma: Standard deviation of normal distribution
            from which observed values are sampled, if ``observed``
            is `True`
        :type sigma: :class:`AuxiliaryParameter`
        """
        super(LogNormalAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size: int):

        self._true_values = np.exp(
            stats.norm.rvs(loc=self.mu, scale=self.tau, size=size))

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)

        else:

            self._obs_values = self._true_values


class Log10NormalAuxSampler(AuxiliarySampler):
    _auxiliary_sampler_name = "Log10NormalAuxSampler"

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A Log10 normal sampler,
        where property ~ 10^N(``mu``, ``sigma``).

        :param name: Name of the property
        :type name: str
        :param observed: `True` if the property is observed,
            `False` if it is latent. Defaults to `True`
        :type observed: bool
        :param mu: Mean of the log10normal
        :type mu: :class:`AuxiliaryParameter`
        :param tau: Standard deviation of the log10normal
        :type tau: :class:`AuxiliaryParameter`
        :param sigma: Standard deviation of normal distribution
            from which observed values are sampled, if ``observed``
            is `True`
        :type sigma: :class:`AuxiliaryParameter`
        """
        super(Log10NormalAuxSampler, self).__init__(name=name,
                                                    observed=observed)

    def true_sampler(self, size: int):

        self._true_values = np.power(
            10, stats.norm.rvs(loc=self.mu, scale=self.tau, size=size))

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)

        else:

            self._obs_values = self._true_values
