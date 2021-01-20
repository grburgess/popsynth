import numpy as np

from popsynth.auxiliary_sampler import NonObservedAuxSampler
from popsynth.utils.spherical_geometry import sample_theta_phi


class SkySampler(object):
    def __init__(self, ra_sampler=None, dec_sampler=None):
        """
        A sky sampler that samples theta phi positions

        :returns:
        :rtype:

        """

        self._ra_sampler = ra_sampler
        self._dec_sampler = dec_sampler

        self._setup_sky()

    def _setup_sky(self):

        if self._ra_sampler is None:
            self._ra_sampler = RASampler()

        if self._dec_sampler is None:
            self._dec_sampler = DecSampler()

    @property
    def ra_sampler(self):

        return self._ra_sampler

    @property
    def dec_sampler(self):

        return self._dec_sampler


class RASampler(NonObservedAuxSampler):
    def __init__(self):
        """

        Samples the RA of sky uniformly

        :param true_values:
        :returns:
        :rtype:

        """

        super(RASampler, self).__init__(name="ra")

    def true_sampler(self, size):

        self._true_values = np.random.uniform(0, 2 * np.pi, size=size)


class DecSampler(NonObservedAuxSampler):
    def __init__(self):
        """
        sampler the dec of the sky uniformly


        :param true_values:
        :returns:
        :rtype:

        """

        super(DecSampler, self).__init__(name="dec")

    def true_sampler(self, size):

        self._true_values = np.arccos(1 - 2 *
                                      np.random.uniform(0.0, 1.0, size=size))
