from popsynth.auxiliary_sampler import AuxiliarySampler
from popsynth.utils.spherical_geometry import sample_theta_phi

class SkySampler(object):
    def __init__(self):

        self._ra_sampler = None
        self._dec_sampler = None

    def _sample_sky(self):

        dec, ra = sample_theta_phi()

        self._dec_sampler = DecSampler(dec)
        self._ra_sampler = RASampler(ra)

    @property
    def ra_sampler(self):

        return self._ra_sampler

    @property
    def dec_sampler(self)
        

class RASampler(AuxiliarySampler):
    def __init__(self, true_values):
        """
        adds the RA values of the sky positions

        :param true_values: 
        :returns: 
        :rtype: 

        """

        self._true_values = true_values

        super(RASampler, self).__init__("ra", sigma=1.0, observed=False, truth={})

    def true_sampler(self, size):

        return self._true_values


class DecSampler(AuxiliarySampler):
    def __init__(self, true_values):
        """
        adds the Dec value of the sky_positions

        :param true_values: 
        :returns: 
        :rtype: 

        """

        self._true_values = true_values

        super(DecSampler, self).__init__("dec", sigma=1.0, observed=False, truth={})

    def true_sampler(self, size):

        return self._true_values
