import numpy as np
import abc

from popsynth.utils.meta import Parameter, ParameterMeta
from popsynth.utils.rejection_sample import rejection_sample
from popsynth.utils.spherical_geometry import sample_theta_phi

from tqdm.autonotebook import tqdm as progress_bar


class DistributionParameter(Parameter):
    pass

class Distribution(object, metaclass=ParameterMeta):
    def __init__(self, name, seed, form):
        """
        A distribution base class

        :param name: the name of the distribution
        :param seed: the random seed
        :param form: the latex form
        :param truth: dictionary holding true parameters
        :returns: 
        :rtype: 

        """

        self._parameter_storage = {}
        
        self._seed = seed
        self._name = name
        self._form = form

    @property
    def name(self):
        return self._name

    @property
    def form(self):
        return self._form

    @property
    def params(self):
        return self._parameter_storage

    @property
    def truth(self):
        return self._parameter_storage


class SpatialDistribution(Distribution):

    r_max = DistributionParameter(vmin=0, default=10)

    def __init__(self, name, seed, form=None):
        """
        A spatial distribution such as a redshift
        distribution

        :param name: the name of the distribution
        :param r_max: the maximum distance to sample
        :param seed: the random seed
        :param form: the latex form
        
        """
        self._theta = None
        self._phi = None

        super(SpatialDistribution, self).__init__(name=name, seed=seed, form=form)

    @abc.abstractmethod
    def differential_volume(self, distance):
        """
        the differential volume

        :param distance: 
        :returns: 
        :rtype: 

        """

        raise RuntimeError("Must be implemented in derived class")

    @abc.abstractmethod
    def dNdV(self, distance):

        raise RuntimeError("Must be implemented in derived class")
        pass

    def time_adjustment(self, r):
        """FIXME! briefly describe function

        :param r:
        :returns:
        :rtype:

        """

        return 1.0

    @abc.abstractmethod
    def transform(self, flux, distance):
        pass

    @property
    def theta(self):
        return self._theta

    @property
    def phi(self):
        return self._phi

    @property
    def distances(self):
        return self._distances

    def draw_sky_positions(self, size):

        self._theta, self._phi = sample_theta_phi(size)

    def draw_distance(self, size, verbose):
        """
        Draw the distances from the specified dN/dr model
        """

        # create a callback for the sampler
        dNdr = (
            lambda r: self.dNdV(r)
            * self.differential_volume(r)
            / self.time_adjustment(r)
        )

        # find the maximum point
        tmp = np.linspace(0.0, self.r_max, 500, dtype=np.float64)
        ymax = np.max(dNdr(tmp))

        # rejection sampling the distribution
        r_out = []

        if verbose:
            for i in progress_bar(range(size), desc="Drawing distances"):
                flag = True
                while flag:

                    # get am rvs from 0 to the max of the function

                    y = np.random.uniform(low=0, high=ymax)

                    # get an rvs from 0 to the maximum distance

                    r = np.random.uniform(low=0, high=self.r_max)

                    # compare them

                    if y < dNdr(r):
                        r_out.append(r)
                        flag = False
        else:

            r_out = rejection_sample(size, ymax, self.r_max, dNdr)

        self._distances = np.array(r_out)


class LuminosityDistribution(Distribution):
    def __init__(self, name, seed, form=None):
        """
        A luminosity distribution such as a 
        distribution

        :param name: the name of the distribution
        :param seed: the random seed
        :param form: the latex form
        """

        super(LuminosityDistribution, self).__init__(
            name=name, seed=seed, form=form,
        )

    @abc.abstractmethod
    def phi(self, luminosity):
        """
        The functional form of the distribution

        :param L: 
        :returns: 
        :rtype: 

        """

        raise RuntimeError("Must be implemented in derived class")

    @abc.abstractmethod
    def draw_luminosity(self, size):
        pass
