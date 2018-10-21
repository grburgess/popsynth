import numpy as np

import abc

class AuxiliarySampler(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, sigma):

        self._sigma = sigma
        
        self._name = name
        self._obs_name = '%s_obs' %name

        self._obs_values = None
        self._true_values = None
        

    def set_luminosity(self, luminosity):

        self._luminosity = luminosity

    def set_distance(self, distance):

        self._distance = distance
        
    @property
    def name(self):
        return self._name

    @property
    def obs_name(self):
        return self._obs_name

    @property
    def sigma(self):
        return self._sigma
    
    @property
    def true_values(self):
        return self._true_values

    @property
    def obs_values(self):
        return self._obs_values
    
    @property
    
    @abc.abstractmethod
    def true_sampler(self, size=1):

        pass

    @abc.abstractmethod
    def observation_sampler(self, size=1):

        pass

    
class DerivedLumAuxSampler(AuxiliarySampler):

    def __init__(self, name, sigma):

        super(DerivedLumAuxSampler, self).__init__(name, sigma)

    def compute_luminosity(self):
        pass
