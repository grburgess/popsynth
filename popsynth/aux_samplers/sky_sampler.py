import numpy as np

from popsynth.auxiliary_sampler import NonObservedAuxSampler
from popsynth.utils.spherical_geometry import sample_theta_phi


class SkySampler(object):
    _auxiliary_sampler_name = "SkySampler"

    def __init__(
        self,
        ra_sampler: NonObservedAuxSampler = None,
        dec_sampler: NonObservedAuxSampler = None,
    ):
        """
        A sky sampler that samples angular positions
        in ra and dec. If no samplers are provided, then
        loads default samplers that sample uniformly on
        the unit sphere. RA and dec are in radians.

        :param ra_sampler: Right ascension (RA) sampler
        :type ra_sampler: :class:`NonObservedAuxSampler`
        :param dec_sampler: Declination (Dec) sampler
        :type dec_sampler: :class:`NonObservedAuxSampler`
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
    _auxiliary_sampler_name = "RASampler"

    def __init__(self):
        """
        Samples the right ascension (RA)
        uniformly on the unit sphere.

        RA is in radians.
        """
        super(RASampler, self).__init__(name="ra")

    def true_sampler(self, size):

        self._true_values = np.random.uniform(0, 2 * np.pi, size=size)


class DecSampler(NonObservedAuxSampler):
    _auxiliary_sampler_name = "DecSampler"

    def __init__(self):
        """
        Samples the declination (Dec)
        uniformly on the unit sphere.

        Dec is in radians.
        """
        super(DecSampler, self).__init__(name="dec")

    def true_sampler(self, size):

        self._true_values = np.arccos(1 - 2 *
                                      np.random.uniform(0.0, 1.0, size=size))
