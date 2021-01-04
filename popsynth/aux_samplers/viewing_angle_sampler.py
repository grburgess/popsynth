import numpy as np

from popsynth.auxiliary_sampler import NonObservedAuxSampler, AuxiliaryParameter


class ViewingAngleSampler(NonObservedAuxSampler):

    max_angle = AuxiliaryParameter(default=90, vmin=0, vmax=180)

    def __init__(self):
        """
        A viewing angle sampler that samples from 0, max_angle.
        It assumes that this is NOT an observed property

        :param max_angle: the maximum angle to which to sample in DEGS
        :returns: None
        :rtype: None

        """

        super(ViewingAngleSampler, self).__init__(name="va", )

    def true_sampler(self, size: int) -> None:
        """
        Sample the viewing angle by inverse CDF

        :param size: number of samples
        :returns: None
        :rtype: None

        """

        theta_inverse = np.random.uniform(0.0,
                                          1 -
                                          np.cos(np.deg2rad(self.max_angle)),
                                          size=size)

        self._true_values = np.arccos(1.0 - theta_inverse)
