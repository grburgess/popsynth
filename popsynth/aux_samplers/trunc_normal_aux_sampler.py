import scipy.stats as stats
import numpy as np

from popsynth.auxiliary_sampler import AuxiliarySampler, AuxiliaryParameter


class TruncatedNormalAuxSampler(AuxiliarySampler):

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    lower = AuxiliaryParameter()
    upper = AuxiliaryParameter()
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name, observed=True):
        """FIXME! briefly describe function

        :param name:
        :param observed:
        :returns:
        :rtype:

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
