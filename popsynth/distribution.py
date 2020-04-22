import numpy as np
import abc


from popsynth.utils.rejection_sample import rejection_sample
from popsynth.utils.spherical_geometry import sample_theta_phi
from tqdm.autonotebook import tqdm as progress_bar


class DistributionParameter(object):
    def __init__(self, default=None, vmin=None, vmax=None):

        self.name = None
        self._vmin = vmin
        self._vmax = vmax
        self._default = default

    @property
    def default(self):
        return self._default

    def __get__(self, obj, type=None) -> object:

        return obj._parameter_storage[self.name]

    def __set__(self, obj, value) -> None:

        if self._vmin is not None:
            assert (
                value >= self._vmin
            ), f"trying to set {self.x} to a value below {self._vmin} is not allowed"

        if self._vmax is not None:
            assert (
                value <= self._vmax
            ), f"trying to set {self.x} to a value above {self._vmax} is not allowed"

        obj._parameter_storage[self.name] = value


class DistributionMeta(type):
    def __new__(mcls, name, bases, attrs, **kwargs):

        if "_parameter_storage" not in attrs:
            attrs["_parameter_storage"] = {}

        cls = super().__new__(mcls, name, bases, attrs, **kwargs)

        # Compute set of abstract method names
        abstracts = {
            name
            for name, value in attrs.items()
            if getattr(value, "__isabstractmethod__", False)
        }
        for base in bases:
            for name in getattr(base, "__abstractmethods__", set()):
                value = getattr(cls, name, None)
                if getattr(value, "__isabstractmethod__", False):
                    abstracts.add(name)
        cls.__abstractmethods__ = frozenset(abstracts)

        for k, v in attrs.items():

            if isinstance(v, DistributionParameter):
                v.name = k

                attrs["_parameter_storage"][k] = v.default

        return cls

    def __subclasscheck__(cls, subclass):
        """Override for issubclass(subclass, cls)."""
        if not isinstance(subclass, type):
            raise TypeError("issubclass() arg 1 must be a class")
        # Check cache

        # Check the subclass hook
        ok = cls.__subclasshook__(subclass)
        if ok is not NotImplemented:
            assert isinstance(ok, bool)
            if ok:
                cls._abc_cache.add(subclass)
            else:
                cls._abc_negative_cache.add(subclass)
            return ok
        # Check if it's a direct subclass
        if cls in getattr(subclass, "__mro__", ()):
            cls._abc_cache.add(subclass)
            return True
        # Check if it's a subclass of a registered class (recursive)
        for rcls in cls._abc_registry:
            if issubclass(subclass, rcls):
                cls._abc_cache.add(subclass)
                return True
        # Check if it's a subclass of a subclass (recursive)
        for scls in cls.__subclasses__():
            if issubclass(subclass, scls):
                cls._abc_cache.add(subclass)
                return True
        # No dice; update negative cache
        cls._abc_negative_cache.add(subclass)
        return False


class Distribution(object, metaclass=DistributionMeta):
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
