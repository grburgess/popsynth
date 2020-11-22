import scipy.stats as stats
import numpy as np
from nptyping import NDArray

from popsynth.auxiliary_sampler import AuxiliarySampler, AuxiliaryParameter


class LogNormalAuxSampler(AuxiliarySampler):

    mu = AuxiliaryParameter(default=0)
    tau = AuxiliaryParameter(default=1, vmin=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A Log normal sampler. None the tru values are in log

        :param name:
        :param observed:
        :returns:
        :rtype:

        """
        super(LogNormalAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size: int):

        self._true_values = stats.norm.rvs(
            loc=np.log10(self.mu), scale=self.tau, size=size
        )  # type: NDArray[np.float64]

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(
                loc=self._true_values, scale=self.sigma, size=size
            )  # type: NDArray[np.float64]

        else:

            self._obs_values = self._true_values  # type: NDArray[np.float64]
