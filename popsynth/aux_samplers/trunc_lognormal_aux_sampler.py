import numpy as np
import scipy.stats as stats

from popsynth.auxiliary_sampler import AuxiliaryParameter, AuxiliarySampler


class TruncatedLogNormalAuxSampler(AuxiliarySampler):
    _auxiliary_sampler_name = "TruncatedLogNormalAuxSampler"

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    lower = AuxiliaryParameter(vmin=0)
    upper = AuxiliaryParameter(vmin=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A truncated Log normal sampler,
        where property ~ e^N(``mu``, ``sigma``), between
        ``lower`` and ``upper``.

        :param name: Name of the property
        :type name: str
        :param observed: `True` if the property is observed,
            `False` if it is latent. Defaults to `True`
        :type observed: bool
        :param mu: Mean of the lognormal
        :type mu: :class:`AuxiliaryParameter`
        :param tau: Standard deviation of the lognormal
        :type tau: :class:`AuxiliaryParameter`
        :param lower: Lower bound of the truncation
        :type lower: :class:`AuxiliaryParameter`
        :param upper: Upper bound of the truncation
        :type upper: :class:`AuxiliaryParameter`
        :param sigma: Standard deviation of normal distribution
            from which observed values are sampled, if ``observed``
            is `True`
        :type sigma: :class:`AuxiliaryParameter`
        """
        super(TruncatedLogNormalAuxSampler, self).__init__(name=name,
                                                           observed=observed)

    def true_sampler(self, size: int):

        if self.lower == 0:
            a = stats.norm.ppf(1e-5, loc=self.mu, scale=self.tau)
        else:
            a = np.log(self.lower)

        b = np.log(self.upper)

        lower = (a - self.mu) / self.tau
        upper = (b - self.mu) / self.tau

        self._true_values = np.exp(
            stats.truncnorm.rvs(
                lower,
                upper,
                loc=self.mu,
                scale=self.tau,
                size=size,
            ))

        assert np.alltrue(self._true_values >= self.lower)
        assert np.alltrue(self._true_values <= self.upper)

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)

        else:

            self._obs_values = self._true_values


class TruncatedLog10NormalAuxSampler(AuxiliarySampler):
    _auxiliary_sampler_name = "TruncatedLog10NormalAuxSampler"

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)
    lower = AuxiliaryParameter(vmin=0)
    upper = AuxiliaryParameter(vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A truncated Log10 normal sampler,
        where property ~ 10^N(``mu``, ``sigma``), between
        ``lower`` and ``upper``.

        :param name: Name of the property
        :type name: str
        :param observed: `True` if the property is observed,
            `False` if it is latent. Defaults to `True`
        :type observed: bool
        :param mu: Mean of the log10normal
        :type mu: :class:`AuxiliaryParameter`
        :param tau: Standard deviation of the log10normal
        :type tau: :class:`AuxiliaryParameter`
        :param lower: Lower bound of the truncation
        :type lower: :class:`AuxiliaryParameter`
        :param upper: Upper bound of the truncation
        :type upper: :class:`AuxiliaryParameter`
        :param sigma: Standard deviation of normal distribution
            from which observed values are sampled, if ``observed``
            is `True`
        :type sigma: :class:`AuxiliaryParameter`
        """
        super(TruncatedLog10NormalAuxSampler, self).__init__(name=name,
                                                             observed=observed)

    def true_sampler(self, size: int):

        if self.lower == 0:
            a = stats.norm.ppf(1e-5, loc=self.mu, scale=self.tau)
        else:
            a = np.log(self.lower)

        b = np.log10(self.upper)

        lower = (a - self.mu) / self.tau
        upper = (b - self.mu) / self.tau

        self._true_values = np.power(
            10,
            stats.truncnorm.rvs(
                lower,
                upper,
                loc=self.mu,
                scale=self.tau,
                size=size,
            ))

        assert np.alltrue(self._true_values >= self.lower)
        assert np.alltrue(self._true_values <= self.upper)

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)

        else:

            self._obs_values = self._true_values
