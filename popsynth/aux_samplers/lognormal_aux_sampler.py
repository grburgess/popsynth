import scipy.stats as stats

from popsynth.auxiliary_sampler import AuxiliarySampler


class LogNormalAuxSampler(AuxiliarySampler):
    def __init__(self, name, mu=0.0, tau=1.0, sigma=None, observed=True):
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

        self._mu = mu
        self._tau = tau

        super(LogNormalAuxSampler, self).__init__(
            name=name, sigma=sigma, observed=observed
        )

    def true_sampler(self, size):

        self._true_values = stats.norm.rvs(loc=np.log10(self._mu), scale=self._tau, size=size)

    def observation_sampler(self, size):

        if self._is_observed:

            self._obs_values = stats.normal.rvs(
                loc=self._true_values, scale=self._sigma
            )

        else:

            self._obs_values = self._true_values
