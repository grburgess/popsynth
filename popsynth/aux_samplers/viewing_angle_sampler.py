import numpy as np

from popsynth.auxiliary_sampler import AuxiliaryParameter, NonObservedAuxSampler


class ViewingAngleSampler(NonObservedAuxSampler):
    _auxiliary_sampler_name = "ViewingAngleSampler"

    max_angle = AuxiliaryParameter(default=90, vmin=0, vmax=180)

    def __init__(self):
        """
        A viewing angle sampler that samples
        from 0 to ``max_angle``. Unlike other samplers,
        it assumes that this is NOT an observed property

        :param max_angle: The maximum angle to which to
            sample in degrees
        :type max_angle: :class:`AuxiliaryParameter`
        """

        super(ViewingAngleSampler, self).__init__(name="va", )

    def true_sampler(self, size: int) -> None:
        """
        Sample the viewing angle by inverse CDF

        :param size: Number of samples
        :type size: int
        """

        theta_inverse = np.random.uniform(0.0,
                                          1 -
                                          np.cos(np.deg2rad(self.max_angle)),
                                          size=size)

        self._true_values = np.arccos(1.0 - theta_inverse)
