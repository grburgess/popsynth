import abc
from typing import Any, Dict, List, Optional

import numpy as np
from class_registry import AutoRegister
from numpy.typing import ArrayLike
from dotmap import DotMap
import pandas as pd
from IPython.display import Markdown, Math, display

from popsynth.distribution import SpatialContainer
from popsynth.selection_probability import SelectionProbability, UnitySelection
from popsynth.utils.cosmology import cosmology
from popsynth.utils.logging import setup_logger
from popsynth.utils.meta import Parameter, ParameterMeta
from popsynth.utils.registry import auxiliary_parameter_registry

log = setup_logger(__name__)

SamplerDict = Dict[str, Dict[str, ArrayLike]]


class SecondaryContainer(object):

    def __init__(
        self,
        name: str,
        true_values: ArrayLike,
        obs_values: ArrayLike,
        selection: ArrayLike,
    ) -> None:
        """
        A container for secondary properties that adds dict
        and dictionary access

        :param name: the name of the secondary
        :type name: str
        :param true_values:
        :type true_values: ArrayLike
        :param obs_values:
        :type obs_values: ArrayLike
        :param selection:
        :type selection: ArrayLike
        :returns:

        """

        self._true_values: ArrayLike = true_values
        self._obs_values: ArrayLike = obs_values
        self._selection: ArrayLike = selection

        self._name: str = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def true_values(self) -> ArrayLike:
        """
        The true (latent) values of the sampler

        :returns:

        """
        return self._true_values

    @property
    def obs_values(self) -> ArrayLike:
        """
        The observed values of the sampler

        :returns:

        """
        return self._obs_values

    @property
    def selection(self) -> ArrayLike:
        """
        The the slection of the values

        :returns:

        """
        return self._selection

    def __getitem__(self, key):

        if key == "selection":
            return self._selection

        elif key == "true_values":
            return self._true_values

        elif key == "obs_values":
            return self._obs_values

        else:

            log.error("trying to access something that does not exist")

            raise RuntimeError()


class SecondaryStorage(DotMap):

    def __init__(self):
        """
        A container for secondary samplers

        :returns:

        """

        super(SecondaryStorage, self).__init__()

    def add_secondary(self, secondary_values: SecondaryContainer) -> None:
        """
        Add on a new secondary

        :param secondary_values:
        :type secondary_values: SecondaryContainer
        :returns:

        """

        self[secondary_values.name] = secondary_values

    def __add__(self, other):

        if self.empty():
            return other

        elif other.empty():

            return self

        else:

            for k, v in other.items():

                self[k] = v

            return self


class AuxiliaryParameter(Parameter):
    pass


class AuxiliarySampler(
        object,
        metaclass=AutoRegister(auxiliary_parameter_registry,
                               base_type=ParameterMeta),
):

    def __init__(
        self,
        name: str,
        observed: bool = True,
        uses_distance: bool = False,
        uses_luminosity: bool = False,
        uses_sky_position: bool = False,
    ) -> None:
        """
        Base class for auxiliary samplers.

        :param name: Name of the sampler
        :type name: str
        :param observed: `True` if the property is observed,
            `False` if it is latent. Defaults to `True`
        :type observed: bool
        :param uses_distance: `True` if sampler uses distance values
        :type uses_distance: bool
        :param uses_luminosity: `True` if sampler uses luminosities
        :type uses_luminosity: bool
        :param uses_sky_position: `True` if sampler uses sky positions
        :type uses_sky_position: bool
        """

        self._parameter_storage = {}  # type: Dict[str, float]
        self._name = name  # type: str
        self._obs_name = "%s_obs" % name  # type: str

        self._obs_values = None  # type: ArrayLike
        self._true_values = None  # type: ArrayLike
        self._is_observed = observed  # type: bool
        self._secondary_samplers = {}  # type: SamplerDict
        self._is_secondary = False  # type: bool
        self._parent_names = []
        self._has_secondary = False  # type: bool
        self._is_sampled = False  # type: bool
        self._selector = UnitySelection()  # type: SelectionProbability
        self._uses_distance = uses_distance  # type: bool
        self._uses_luminosity = uses_luminosity  # type: bool
        self._uses_sky_position = uses_sky_position  # type: bool

    def display(self):

        out = {"parameter": [], "value": []}

        for k, v in self._parameter_storage.items():

            out["parameter"].append(k)
            out["value"].append(v)

        display(pd.DataFrame(out))

    def __repr__(self):

        out = f"{self._name}\n"

        out += f"observed: {self._is_observed}\n"

        for k, v in self._parameter_storage.items():
            out += f"{k}: {v}\n"

        if self._is_secondary:
            out += f"parents: {self._parent_names}\n"

        if self._has_secondary:

            for k, v in self._secondary_samplers.items():

                out += f"{k}\n"

        return out

    def set_luminosity(self, luminosity: ArrayLike) -> None:
        """
        Set the luminosity values.

        :param luminosity: Luminosity
        :type luminosity: ArrayLike
        """

        self._luminosity = luminosity  # type:ArrayLike

    def set_spatial_values(self, value: SpatialContainer) -> None:
        """
        Set the spatial values.

        :param value: Spatial values
        :type value: :class:`SpatialContainer`
        """

        self._distance = value.distance  # type:ArrayLike
        self._theta = value.theta
        self._phi = value.phi
        self._ra = value.ra
        self._dec = value.dec
        self._spatial_values = value

    def set_selection_probability(self,
                                  selector: SelectionProbability) -> None:
        """
        Set a selection probabilty for this sampler.

        :param selector: A selection probability oobject
        :type selector: SelectionProbability
        :returns:

        """
        if not isinstance(selector, SelectionProbability):

            log.error("The selector is not a valid selection probability")

            raise AssertionError()

        self._selector = selector  # type: SelectionProbability

    def _apply_selection(self) -> None:
        """
        Default selection if none is specfied in child class.
        """

        self._selector.draw(len(self._obs_values))

    def set_secondary_sampler(self, sampler: "AuxiliarySampler") -> None:
        """
        Add a secondary sampler upon which this sampler will depend.
        The sampled values can be accessed via an internal dictionary
        with the samplers 'name'

        self._secondary_sampler['name'].true_values
        self._secondary_sampler['name'].obs_values

        :param sampler: An auxiliary sampler
        :type sampler: "AuxiliarySampler"
        :returns:

        """

        # make sure we set the sampler as a secondary
        # this causes it to throw a flag in the main
        # loop if we try to add it again

        sampler.make_secondary(self.name)
        # attach the sampler to this class

        self._secondary_samplers[sampler.name] = sampler
        self._has_secondary = True  # type: bool

    def draw(self, size: int = 1):
        """
        Draw the primary and secondary samplers. This is the main call.

        :param size: The number of samples to draw
        :type size: int
        """
        # do not resample!
        if not self._is_sampled:

            log.info(f"Sampling: {self.name}")

            if self._has_secondary:

                log.info(f"{self.name} is sampling its secondary quantities")

            self._selector.set_distance(self._distance)

            try:

                self._selector.set_luminosity(self._luminosity)

            except (AttributeError):

                log.debug("tried to set luminosity, but could not")

                pass

            for k, v in self._secondary_samplers.items():

                if not v.is_secondary:

                    log.error("Tried to sample a non-secondary, this is a bug")

                    raise RuntimeError()

                # we do not allow for the secondary
                # quantities to derive a luminosity
                # as it should be the last thing dervied

                log.debug(f"{k} will have it spatial values set")

                v.set_spatial_values(self._spatial_values)

                v.draw(size=size)

            # Now, it is assumed that if this sampler depends on the previous samplers,
            # then those properties have been drawn

            self.true_sampler(size=size)

            if self._is_observed:

                log.debug(f"{self.name} is sampling the observed values")

                self.observation_sampler(size)

            else:

                self._obs_values = self._true_values  # type: ArrayLike

            self._selector.set_observed_value(self._obs_values)

            # check to make sure we sampled!
            assert (self.true_values is not None
                    and len(self.true_values) == size
                    ), f"{self.name} likely has a bad true_sampler function"
            assert (self.obs_values is not None
                    and len(self.obs_values) == size
                    ), f"{self.name} likely has a observation_sampler function"

            # now apply the selection to yourself
            # if there is nothing coded, it will be
            # list of all true

            self._is_sampled = True

            self._apply_selection()

    def reset(self):
        """
        Reset all the selections.
        """

        if self._is_sampled:

            log.info(f"Auxiliary sampler: {self.name} is being reset")

            self._is_sampled = False
            self._obs_values = None  # type: ArrayLike
            self._true_values = None  # type: ArrayLike

            self._selector.reset()

        else:

            log.debug(
                f"{self.name} is not reseting as it has not been sampled")

        for k, v in self._secondary_samplers.items():

            v.reset()

    def make_secondary(self, parent_name: str) -> None:
        """
        sets this sampler as secondary for book keeping

        :param parent_name:
        :type parent_name: str
        :returns:

        """
        self._is_secondary = True  # type: bool
        self._parent_names.append(parent_name)

    def get_secondary_properties(
        self,
        graph=None,
        primary=None,
        spatial_distribution=None,
    ) -> SecondaryStorage:
        """
        Get properties of secondary samplers.

        :param graph: Graph
        :param primary: Primary sampler
        :param spatial_distribution: Spatial Distribution
        :returns: Dict of samplers
        :rtype: :class:`SamplerDict`
        """

        recursive_secondaries: SecondaryStorage = SecondaryStorage()

        # now collect each property. This should keep recursing
        if self._has_secondary:

            for k, v in self._secondary_samplers.items():

                if graph is not None:

                    graph.add_node(k, observed=False)
                    graph.add_edge(k, primary)

                    if v.observed:
                        graph.add_node(v.obs_name, observed=False)
                        graph.add_edge(k, v.obs_name)

                    if v.uses_distance:

                        self._graph.add_edge(spatial_distribution.name, k)

                recursive_secondaries += v.get_secondary_properties(
                    graph, k, spatial_distribution)

        # add our own on

        recursive_secondaries.add_secondary(
            SecondaryContainer(self._name, self._true_values, self._obs_values,
                               self._selector))

        return recursive_secondaries

    def get_secondary_objects(
        self,
        recursive_secondaries: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Get secondary objects.

        :param recursive_secondaries: Recursive dict of secondaries
        :returns: Dict of objects
        :rtype: Dict[str, Any]
        """

        # if a holder was not passed, create one
        if recursive_secondaries is None:

            recursive_secondaries = {}  # type: SamplerDict

        # now collect each property. This should keep recursing
        if self._has_secondary:

            for k, v in self._secondary_samplers.items():
                recursive_secondaries = v.get_secondary_objects(
                    recursive_secondaries)  # type: SamplerDict

        # add our own on

        tmp = {}

        tmp["type"] = self._auxiliary_sampler_name
        tmp["observed"] = self.observed

        for k2, v2 in self.truth.items():

            tmp[k2] = v2

        tmp["secondary"] = list(self.secondary_samplers.keys())

        selection = {}
        selection[self.selector._selection_name] = self.selector.parameters

        tmp["selection"] = selection

        recursive_secondaries[self._name] = tmp

        return recursive_secondaries

    @property
    def secondary_samplers(self) -> SamplerDict:
        """
        Secondary samplers.
        :returns: Dict of secondary samplers
        :rtype: :class:`SamplerDict`
        """

        return self._secondary_samplers

    @property
    def is_secondary(self) -> bool:
        """

        If another sampler depends on this

        :returns:

        """
        return self._is_secondary

    @property
    def parents(self) -> List[str]:
        """
        The parents of this sampler
        """
        return self._parent_names

    @property
    def has_secondary(self) -> bool:
        """
        if this sampler has a secondary
        :returns:

        """
        return self._has_secondary

    @property
    def observed(self) -> bool:
        """
        if this sampler is observed

        :returns:

        """

        return self._is_observed

    @property
    def name(self) -> str:
        """
        The name of the sampler

        :returns:

        """
        return self._name

    @property
    def obs_name(self) -> str:
        return self._obs_name

    @property
    def true_values(self) -> np.ndarray:
        """
        The true or latent values

        :returns:

        """
        return self._true_values

    @property
    def obs_values(self) -> np.ndarray:
        """
        The values obscured by measurement error.

        :returns:

        """
        return self._obs_values

    @property
    def selection(self) -> np.ndarray:
        """
        The selection booleans on the values

        :returns:

        """
        return self._selector.selection

    @property
    def selector(self) -> SelectionProbability:
        """
        The selection probability object

        :returns:

        """
        return self._selector

    @property
    def truth(self) -> Dict[str, float]:
        """
        A dictionary containing true paramters
        used to simulate the distribution
        """
        out = {}

        for k, v in self._parameter_storage.items():

            if v is not None:

                out[k] = v

        return out

    @property
    def uses_distance(self) -> bool:
        """
        If this uses distance

        :returns:

        """
        return self._uses_distance

    @property
    def uses_sky_position(self) -> bool:
        """
        If this uses sky position

        :returns:

        """
        return self._uses_sky_position

    @property
    def uses_luminosity(self) -> np.ndarray:
        """
        If this uses luminosity

        :returns:

        """
        return self._uses_luminosity

    @property
    def luminosity_distance(self):
        """
        luminosity distance if needed.
        """

        return cosmology.luminosity_distance(self._distance)

    @abc.abstractmethod
    def true_sampler(self, size: int = 1):

        pass

    def observation_sampler(self, size: int = 1) -> np.ndarray:

        return self._true_values


class NonObservedAuxSampler(AuxiliarySampler):

    def __init__(self,
                 name: str,
                 uses_distance: bool = False,
                 uses_luminosity: bool = False):

        super(NonObservedAuxSampler, self).__init__(
            name=name,
            observed=False,
            uses_distance=uses_distance,
            uses_luminosity=uses_luminosity,
        )


class DerivedLumAuxSampler(AuxiliarySampler):

    def __init__(self, name: str, uses_distance: bool = False):
        """
        Base class for generating luminosity from other properties.

        :param name: Name of the sampler
        :type name: str
        :param uses_distance: `True` if sampler uses distance values
        :type uses_distance: bool
        """

        super(DerivedLumAuxSampler, self).__init__(name,
                                                   observed=False,
                                                   uses_distance=uses_distance)

    @abc.abstractmethod
    def compute_luminosity(self):

        raise RuntimeError("Must be implemented in derived class")
