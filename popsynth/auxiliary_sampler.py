import numpy as np

import abc


class AuxiliarySampler(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, sigma, observed=True):

        self._sigma = sigma

        self._name = name
        self._obs_name = '%s_obs' % name

        self._obs_values = None
        self._true_values = None
        self._is_observed = observed
        self._secondary_samplers = {}
        self._is_secondary = False

    def set_luminosity(self, luminosity):
        """FIXME! briefly describe function

        :param luminosity: 
        :returns: 
        :rtype: 

        """

        self._luminosity = luminosity

    def set_distance(self, distance):
        """FIXME! briefly describe function

        :param distance: 
        :returns: 
        :rtype: 

        """
        

        self._distance = distance

    def set_secondary_sampler(self, sampler):
        """
        Allows the setting of a secondary sampler from which to derive values
        """

        # make sure we set the sampler as a secondary
        # this causes it to throw a flag in the main
        # loop if we try to add it again
        
        sampler.make_secondary()

        # attach the sampler to this class
        
        self._secondary_samplers[sampler.name] = sampler

    def draw(self, size=1):
        """
        Draw the primary and secondary samplers. This is the main call.

        :param size: the number of samples to draw
        """

        print("%s is sampling its secondary quantities" % self.name)

        for k, v in self._secondary_samplers.items():

            print('Sampling: %s' % k)

            # we do not allow for the secondary
            # quantities to derive a luminosity
            # as it should be the last thing dervied

            v.draw(size=size)

        # Now, it is assumed that if this sampler depends on the previous samplers,
        # then those properties have been drawn

        self.true_sampler(size=size)

        if self._is_observed:

            self.observation_sampler(size)

        else:

            self._obs_values = np.array([-9999999] * size)

        # check to make sure we sampled!
        assert self.true_values is not None and len(self.true_values) == size
        assert self.obs_values is not None and len(self.obs_values) == size

    def make_secondary(self):

        self._is_secondary = True

    @property
    def secondary_samplers(self):
        """
        Secondary samplers
        """

        return self._secondary_samplers

    @property
    def is_secondary(self):

        return self._is_secondary

    @property
    def observed(self):
        """
        """
        return self._is_observed

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

    def __init__(self, name, sigma, observed=True):
        """FIXME! briefly describe function

        :param name: 
        :param sigma: 
        :param observed: 
        :returns: 
        :rtype: 

        """
        

        super(DerivedLumAuxSampler, self).__init__(name, sigma)

    def compute_luminosity(self):

        raise RuntimeError('Must be implemented in derived class')
