import numpy as np
import abc


from popsynth.utils.rejection_sample import rejection_sample

from tqdm.autonotebook import tqdm as progress_bar


class Distribution(object):
    def __init__(self, name, seed, form, truth=None):
        """FIXME! briefly describe function

        :param name: 
        :param seed: 
        :param form: 
        :param truth: 
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
        """FIXME! briefly describe function

        :param name: 
        :param r_max: 
        :param seed: 
        :param form: 
        :param truth: 
        :returns: 
        :rtype: 

        """

        self._r_max = r_max

        super(SpatialDistribution, self).__init__(
            name=name, seed=seed, form=form, truth=truth
        )

    @abc.abstractmethod
    def differential_volume(self, distance):

        raise RuntimeError("Must be implemented in derived class")
        pass

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

        return np.array(r_out)

    @property
    def r_max(self):
        return self._r_max


class LuminosityDistribution(Distribution):
    __metaclass__ = abc.ABCMeta

    def __init__(self, name, seed, form=None, truth=None):
        """FIXME! briefly describe function

        :param name: 
        :param seed: 
        :param form: 
        :param truth: 
        :returns: 
        :rtype: 

        """

        super(LuminosityDistribution, self).__init__(
            name=name, seed=seed, form=form, truth=truth
        )

        self._params = None

    @abc.abstractmethod
    def phi(self, L):

        raise RuntimeError("Must be implemented in derived class")

        pass

    @abc.abstractmethod
    def draw_luminosity(self, size):
        pass
