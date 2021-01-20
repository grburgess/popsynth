#from numpy.typing import ArrayLike
from typing import List

import numpy as np
import scipy.stats as stats

from popsynth.auxiliary_sampler import AuxiliaryParameter, AuxiliarySampler

ArrayLike = List[float]


class LogNormalAuxSampler(AuxiliarySampler):

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A Log normal sampler

        where x ~ 10^N(mu, sigma)


        :param name:
        :param observed:
        :returns:
        :rtype:

        """
        super(LogNormalAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size: int):

        self._true_values = np.exp(
            stats.norm.rvs(loc=self.mu, scale=self.tau,
                           size=size))  # type: ArrayLike

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)  # type: ArrayLike

        else:

            self._obs_values = self._true_values  # type: ArrayLike


class Log10NormalAuxSampler(AuxiliarySampler):

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A Log10 normal sampler.

        where x ~ 10^N(mu, sigma)



        :param name:
        :param observed:
        :returns:
        :rtype:

        """
        super(Log10NormalAuxSampler, self).__init__(name=name,
                                                    observed=observed)

    def true_sampler(self, size: int):

        self._true_values = np.power(10,
                                     stats.norm.rvs(
                                         loc=self.mu,
                                         scale=self.tau,
                                         size=size))  # type: ArrayLike

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)  # type: ArrayLike

        else:

            self._obs_values = self._true_values  # type: ArrayLike
