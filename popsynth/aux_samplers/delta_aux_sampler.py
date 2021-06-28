import numpy as np
import scipy.stats as stats

from popsynth.auxiliary_sampler import AuxiliarySampler, AuxiliaryParameter


class DeltaAuxSampler(AuxiliarySampler):
    _auxiliary_sampler_name = "DeltaAuxSampler"

    xp = AuxiliaryParameter(default=0)
    sigma = AuxiliaryParameter(default=1, vmin=0)

    def __init__(self, name: str, observed: bool = True):
        """
        A delta-function sampler for which the true value
        is fixed at ``xp``. Assumes property is observed by default,
        in which case the observed value is sampled from
        the true value with some normally-distributed error, ``sigma``.

        :param name: Name of the property
        :type name: str
        :param observed: `True` if the property is observed,
            `False` if it is latent. Defaults to `True`
        :type observed: bool
        :param xp: Value at which delta function is located
        :type xp: :class:`AuxiliaryParameter`
        :param sigma: Standard deviation of normal distribution
            from which observed values are sampled, if ``observed``
            is `True`
        :type sigma: :class:`AuxiliaryParameter`
        """

        super(DeltaAuxSampler, self).__init__(name=name, observed=observed)

    def true_sampler(self, size: int):

        self._true_values = np.repeat(self.xp, repeats=size)

    def observation_sampler(self, size: int):

        if self._is_observed:

            self._obs_values = stats.norm.rvs(loc=self._true_values,
                                              scale=self.sigma,
                                              size=size)

        else:

            self._obs_values = self._true_values
