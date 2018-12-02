import numpy as np

from popsynth.auxiliary_sampler import AuxiliarySampler


class ViewingAngleSampler(AuxiliarySampler):
    
    def __init__(self,max_angle=90.):
        """ 
        A viewing angle sampler that samples from 0, max_angle.
        It assumes that this is NOT an observed property

        :param max_angle: the maximum angle to which to sample in DEGS
        :returns: None
        :rtype: None

        """
        
        assert (max_angle> 0.) and (max_angle<=90), 'angle must be between 0 and 90.'
        
        self._max_angle = np.deg2rad(max_angle)
        
        super(ViewingAngleSampler, self).__init__('va', sigma=1., observed=False)
        
    def true_sampler(self, size):
        """
        Sample the viewing angle by inverse CDF 

        :param size: number of samples
        :returns: None
        :rtype: None

        """
        
        theta_inverse = np.random.uniform(0., 1-np.cos(self._max_angle), size=size )
        
        self._true_values =  np.arccos(1.-theta_inverse)
