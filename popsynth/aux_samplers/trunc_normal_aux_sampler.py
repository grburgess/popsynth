import scipy.stats as stats
import numpy as np

from popsynth.auxiliary_sampler import AuxiliarySampler, AuxiliaryParameter


class TruncatedNormalAuxSampler(AuxiliarySampler):
    _auxiliary_sampler_name = "TruncatedNormalAuxSampler"

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    lower = AuxiliaryParameter()
    upper = AuxiliaryParameter()
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A truncated normal sampler,
        where property ~ N(``mu``, ``sigma``), between
        ``lower`` and ``upper``.

        :param name: Name of the property
        :type name: str
        :param observed: `True` if the property is observed,
            `False` if it is latent. Defaults to `True`
        :type observed: bool
        :param mu: Mean of the normal
        :type mu: :class:`AuxiliaryParameter`
        :param tau: Standard deviation of the normal
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

        super(TruncatedNormalAuxSampler, self).__init__(name=name,
                                                        observed=observed)

    def true_sampler(self, size):

        l = (self.lower - self.mu) / self.tau
        u = (self.upper - self.mu) / self.tau

        self._true_values = stats.truncnorm.rvs(
            l,
            u,
            loc=self.mu,
            scale=self.tau,
            size=size,
        )

        assert np.alltrue(self._true_values >= self.lower)
        assert np.alltrue(self._true_values <= self.upper)

    def observation_sampler(self, size):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)

        else:

            self._obs_values = self._true_values
