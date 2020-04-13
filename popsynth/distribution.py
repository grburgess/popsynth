import numpy as np
import abc


from popsynth.utils.rejection_sample import rejection_sample
from popsynth.utils.spherical_geometry import sample_theta_phi
from tqdm.autonotebook import tqdm as progress_bar


class Distribution(object):
    def __init__(self, name, seed, form, truth=None):
        """
        A distribution base class

        :param name: the name of the distribution
        :param seed: the random seed
        :param form: the latex form
        :param truth: dictionary holding true parameters
        :returns: 
        :rtype: 

        """
        self._seed = seed
        self._name = name
        self._form = form

        if truth is None:
            self._truth = {}

        else:

            self._truth = truth

        # construct the params
        # just in case there are none
        self._construct_distribution_params()

    @property
    def name(self):
        return self._name

    @property
    def form(self):
        return self._form

    @property
    def params(self):
        return self._params

    @property
    def truth(self):
        return self._truth

    def _construct_distribution_params(self, **params):
        """
        Build the initial distributional parameters
        """

        self._params = {}

        for k, v in params.items():

            self._params[k] = v

    def set_distribution_params(self, **params):
        """
        Set the spatial parameters as keywords
        """

        try:

            for k, v in params.items():

                if k in self._params:
                    self._params[k] = v
                else:
                    RuntimeWarning(
                        "%s was not originally in the parameters... ignoring." % k
                    )

        except:

            # we have not set these before

            self._params = params


class SpatialDistribution(Distribution):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, r_max, seed, form=None, truth=None):
        """
        A spatial distribution such as a redshift
        distribution

        :param name: the name of the distribution
        :param r_max: the maximum distance to sample
        :param seed: the random seed
        :param form: the latex form
        :param truth: the true parameter dictionary
        
        """

        self._r_max = r_max

        self._theta = None
        self._phi = None

        super(SpatialDistribution, self).__init__(
            name=name, seed=seed, form=form, truth=truth
        )

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
        tmp = np.linspace(0.0, self._r_max, 500, dtype=np.float64)
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

                    r = np.random.uniform(low=0, high=self._r_max)

                    # compare them

                    if y < dNdr(r):
                        r_out.append(r)
                        flag = False
        else:

            r_out = rejection_sample(size, ymax, self._r_max, dNdr)

        self._distances = np.array(r_out)

        return np.array(r_out)

    @property
    def r_max(self):
        return self._r_max


class LuminosityDistribution(Distribution):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, seed, form=None, truth=None):
        """
        A luminosity distribution such as a 
        distribution

        :param name: the name of the distribution
        :param seed: the random seed
        :param form: the latex form
        :param truth: the true parameter dictionary

        """

        super(LuminosityDistribution, self).__init__(
            name=name, seed=seed, form=form, truth=truth
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
