import scipy.stats as stats
import numpy as np

from popsynth.auxiliary_sampler import AuxiliarySampler, AuxiliaryParameter


class LogNormalAuxSampler(AuxiliarySampler):

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name, observed=True):
        """
        A Log normal sampler. None the tru values are in log

        :param name: 
        :param mu: 
        :param tau: 
        :param sigma: 
        :param observed: 
        :returns: 
        :rtype: 

        """
        super(LogNormalAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size):

        self._true_values = stats.norm.rvs(
            loc=np.log10(self.mu), scale=self.tau, size=size
        )

    def observation_sampler(self, size):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(
                loc=self._true_values, scale=self.sigma, size=size
            )

        else:

            self._obs_values = self._true_values
