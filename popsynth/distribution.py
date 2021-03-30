import abc
from dataclasses import dataclass
from typing import Dict, List, Union

import numpy as np

from popsynth.utils.configuration import popsynth_config
from popsynth.utils.logging import setup_logger
from popsynth.utils.meta import Parameter, ParameterMeta
from popsynth.utils.progress_bar import progress_bar
from popsynth.utils.rejection_sample import rejection_sample
from popsynth.utils.spherical_geometry import sample_theta_phi

#from numpy.typing import ArrayLike

ArrayLike = List[float]


log = setup_logger(__name__)


class DistributionParameter(Parameter):
    pass


class Distribution(object, metaclass=ParameterMeta):
    def __init__(self, name: str, seed: int, form: str) -> None:
        """
        A distribution base class

        :param name: the name of the distribution
        :param seed: the random seed
        :param form: the latex form
        :param truth: dictionary holding true parameters
        :returns:
        :rtype:

        """

        self._parameter_storage = {}  # type: dict

        self._seed = seed  # type: int
        self._name = name  # type: str
        self._form = form  # type: str

    @property
    def name(self) -> str:
        return self._name

    @property
    def form(self) -> str:
        return self._form

    @property
    def params(self) -> Dict[str, float]:
        return self._parameter_storage

    @property
    def truth(self) -> Dict[str, float]:
        return self._parameter_storage


@dataclass
class SpatialContainer:

    distance: ArrayLike
    theta: ArrayLike
    phi: ArrayLike

    @property
    def dec(self) -> np.ndarray:
        return 90 - np.rad2deg(self.theta)

    @property
    def ra(self) -> np.ndarray:
        return np.rad2deg(self.phi)

        

    
class SpatialDistribution(Distribution):

    r_max = DistributionParameter(vmin=0, default=10)

    def __init__(self, name: str, seed: int, form: Union[str, None] = None):
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

        super(SpatialDistribution, self).__init__(name=name,
                                                  seed=seed,
                                                  form=form)

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
    def theta(self) -> np.ndarray:
        return self._theta

    @property
    def phi(self) -> np.ndarray:
        return self._phi

    @property
    def dec(self) -> np.ndarray:
        return 90 - np.rad2deg(self._theta)

    @property
    def ra(self) -> np.ndarray:
        return np.rad2deg(self._phi)

    @property
    def distances(self) -> np.ndarray:
        return self._distances

    @property
    def spatial_values(self):

        return SpatialContainer(self._distances, self._theta, self._phi)
    
    def draw_sky_positions(self, size: int) -> None:

        self._theta, self._phi = sample_theta_phi(size)

    def draw_distance(self, size: int) -> None:
        """
        Draw the distances from the specified dN/dr model
        """

        # create a callback for the sampler
        dNdr = (lambda r: self.dNdV(r) * self.differential_volume(r) / self.
                time_adjustment(r))

        # find the maximum point
        tmp = np.linspace(0.0, self.r_max, 500,
                          dtype=np.float64)  # type: ArrayLike
        ymax = np.max(dNdr(tmp))  # type: float

        # rejection sampling the distribution
        r_out = []

        # slow

        if popsynth_config["show_progress"]:
            for i in progress_bar(range(size), desc="Drawing distances"):
                flag = True
                while flag:

                    # get am rvs from 0 to the max of the function

                    y = np.random.uniform(low=0, high=ymax)  # type: float

                    # get an rvs from 0 to the maximum distance

                    r = np.random.uniform(low=0,
                                          high=self.r_max)  # type: float

                    # compare them

                    if y < dNdr(r):
                        r_out.append(r)
                        flag = False
        else:

            r_out = rejection_sample(size, ymax, self.r_max,
                                     dNdr)  # type: ArrayLike

        self._distances = np.array(r_out)  # type: ArrayLike


class LuminosityDistribution(Distribution):
    def __init__(self, name: str, seed: int, form: Union[str, None] = None):
        """
        A luminosity distribution such as a
        distribution

        :param name: the name of the distribution
        :param seed: the random seed
        :param form: the latex form
        """

        super(LuminosityDistribution, self).__init__(
            name=name,
            seed=seed,
            form=form,
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
