import abc
from dataclasses import dataclass
from typing import Dict, Union

import numpy as np
from class_registry import AutoRegister
from numpy.typing import ArrayLike
from IPython.display import Markdown, Math, display
import pandas as pd

from popsynth.utils.configuration import popsynth_config
from popsynth.utils.logging import setup_logger
from popsynth.utils.meta import Parameter, ParameterMeta
from popsynth.utils.progress_bar import progress_bar
from popsynth.utils.registry import distribution_registry
from popsynth.utils.rejection_sample import rejection_sample
from popsynth.utils.spherical_geometry import sample_theta_phi

log = setup_logger(__name__)


class DistributionParameter(Parameter):
    pass


class Distribution(object,
                   metaclass=AutoRegister(distribution_registry,
                                          base_type=ParameterMeta)):
    _distribution_name = "Distribution"

    def __init__(self, name: str, seed: int, form: str) -> None:
        """
        A distribution base class.

        :param name: Name of the distribution
        :type name: str
        :param seed: Random seed
        :type seed: int
        :param form: the LaTeX form
        :type form: str
        """

        self._parameter_storage: Dict[str, float] = {}

        self._seed = seed  # type: int
        self._name = name  # type: str
        self._form = form  # type: str

    @property
    def name(self) -> str:
        """
        The name of the distribution

        :returns: 

        """
        return self._name

    @property
    def form(self) -> str:
        """
        The latex form of the 
        distribution

        :returns: 

        """
        return self._form

    @property
    def params(self) -> Dict[str, float]:
        """
        The parameters of the distribution

        """
        return self._parameter_storage

    @property
    def truth(self) -> Dict[str, float]:
        """
        value of the parameters used to simulate
        """
        out = {}

        for k, v in self._parameter_storage.items():

            if v is not None:

                out[k] = v

        return out

    def display(self):
        """
        use ipython pretty display to 
        display the functions

        :returns: 

        """

        out = {"parameter": [], "value": []}

        for k, v in self.params.items():

            out["parameter"].append(k)
            out["value"].append(v)

        display(Math(self._form))
        display(pd.DataFrame(out))

    def __repr__(self):

        out = f"{self._name}\n"
        out += f"{self._form}\n"

        for k, v in self.params.items():
            out += f"{k}: {v}\n"

        return out


@dataclass
class SpatialContainer:
    """
    Container for 3D spatial values.
    """

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
    _distribution_name = "SpatialDistribution"

    r_max = DistributionParameter(vmin=0, default=10)

    def __init__(self, name: str, seed: int, form: Union[str, None] = None):
        """
        A base class for spatial distributions,
        such as redshift distributions.

        :param name: Name of the distribution
        :type name: str
        :param seed: Random seed
        :type seed: int
        :param form: the LaTeX form
        :type form: Union[str, None]
        """
        self._theta = None
        self._phi = None

        super(SpatialDistribution, self).__init__(name=name,
                                                  seed=seed,
                                                  form=form)

    @abc.abstractmethod
    def differential_volume(self, distance):
        """
        The differential volume

        :param distance: Distance
        """

        raise RuntimeError("Must be implemented in derived class")

    @abc.abstractmethod
    def dNdV(self, distance):
        """
        The differential number of objects
        per volume element

        :param distance: 
        :type distance: 
        :returns: 

        """
        raise RuntimeError("Must be implemented in derived class")
        pass

    def time_adjustment(self, distance):
        """
        The time adjustment

        :param distance: Distance
        """

        return 1.0

    @abc.abstractmethod
    def transform(self, flux, distance):
        """
        The transform from luminosity to flux
        for the 

        :param flux: 
        :type flux: 
        :param distance: 
        :type distance: 
        :returns: 

        """
        pass

    @property
    def theta(self) -> np.ndarray:
        """
        the polar coordinate of the objects

        :returns: 

        """
        return self._theta

    @property
    def phi(self) -> np.ndarray:
        """
        the longitudinal coordinate fo the objects

        :returns: 

        """

        return self._phi

    @property
    def dec(self) -> np.ndarray:
        """
        The declination of the objects

        :returns: 

        """
        return 90 - np.rad2deg(self._theta)

    @property
    def ra(self) -> np.ndarray:
        """
        the right acension of the objects

        :returns: 

        """
        return np.rad2deg(self._phi)

    @property
    def distances(self) -> np.ndarray:
        """
        the distances to the objects

        :returns: 

        """
        return self._distances

    @property
    def spatial_values(self):
        """
        All the spatial values of the objects
        :returns: 

        """
        return SpatialContainer(self._distances, self._theta, self._phi)

    def draw_sky_positions(self, size: int) -> None:
        """
        Draw teh sky positions of the objects

        :param size: 
        :type size: int
        :returns: 

        """
        self._theta, self._phi = sample_theta_phi(size)

    def draw_distance(self, size: int) -> None:
        """
        Draw the distances from the specified dN/dr model.

        :param size: Number of distances to sample
        :type size: int
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

        if popsynth_config.show_progress:
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
    _distribution_name = "LuminosityDistribtuion"

    def __init__(self, name: str, seed: int, form: Union[str, None] = None):
        """
        A base class for luminosity distributions.

        :param name: Name of the distribution
        :type name: str
        :param seed: Random seed
        :type seed: int
        :param form: the LaTeX form
        :type form: Union[str, None]
        """

        super(LuminosityDistribution, self).__init__(
            name=name,
            seed=seed,
            form=form,
        )

    @abc.abstractmethod
    def phi(self, luminosity):
        """
        The functional form of the distribution.
        not required for sampling
        :param luminosity: Luminosity
        """

        raise RuntimeError("Must be implemented in derived class")

    @abc.abstractmethod
    def draw_luminosity(self, size):
        """
        function to draw the luminosity via an alternative method
        must be implemented in child class

        :param size: 
        :type size: 
        :returns: 

        """
        pass
