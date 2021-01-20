import scipy.stats as stats

from popsynth.auxiliary_sampler import AuxiliarySampler, AuxiliaryParameter


class NormalAuxSampler(AuxiliarySampler):

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """FIXME! briefly describe function

        :param name:
        :param mu:
        :param tau:
        :param sigma:
        :param observed:
        :returns:
        :rtype:

        """

        super(NormalAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size: int):

        self._true_values = stats.norm.rvs(loc=self.mu,
                                           scale=self.tau,
                                           size=size)

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)

        else:

            self._obs_values = self._true_values
