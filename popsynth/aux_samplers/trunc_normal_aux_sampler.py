import scipy.stats as stats

from popsynth.auxiliary_sampler import AuxiliarySampler


class TruncatedNormalAuxSampler(AuxiliarySampler):
    def __init__(self, name, lower, upper, mu=0.0, tau=1.0, sigma=None, observed=True):
        """FIXME! briefly describe function

        :param name: 
        :param mu: 
        :param tau: 
        :param sigma: 
        :param observed: 
        :returns: 
        :rtype: 

        """

        self._mu = mu
        self._tau = tau
        self._lower = lower
        self._upper = upper

        super(TruncatedNormalAuxSampler, self).__init__(
            name=name, sigma=sigma, observed=observed
        )

    def true_sampler(self, size):

        self._true_values = stats.truncnorm.rvs(
            (self._lower - self._mu) / self._tau,
            (self._upper - self._mu) / self._tau,
            loc=self._mu,
            scale=self._tau,
            size=size,
        )

    def observation_sampler(self, size):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(
                loc=self._true_values, scale=self._sigma, size=size
            )

        else:

            self._obs_values = self._true_values
