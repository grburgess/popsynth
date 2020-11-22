import abc
from typing import Any, Dict, Union

import numpy as np
from nptyping import NDArray

from popsynth.selection_probability import SelectionProbabilty, UnitySelection
from popsynth.utils.meta import Parameter, ParameterMeta

SamplerDict = Dict[str, Dict[str, NDArray[Any]]]


class AuxiliaryParameter(Parameter):
    pass


class AuxiliarySampler(object, metaclass=ParameterMeta):
    def __init__(
        self,
        name: str,
        observed: bool = True,
        uses_distance: bool = False,
        uses_luminosity: bool = False,
    ) -> None:

        self._parameter_storage = {}  # type: Dict[str, float]
        self._name = name  # type: str
        self._obs_name = "%s_obs" % name  # type: str

        self._obs_values = None  # type: NDArray[Any]
        self._true_values = None  # type: NDArray[Any]
        self._is_observed = observed  # type: bool
        self._secondary_samplers = {}  # type: SamplerDict
        self._is_secondary = False  # type: bool
        self._has_secondary = False  # type: bool
        self._is_sampled = False  # type: bool
        self._selector = UnitySelection()  # type: SelectionProbabilty
        self._uses_distance = uses_distance  # type: bool
        self._uses_luminoity = uses_luminosity  # type: bool

    def set_luminosity(self, luminosity: NDArray[np.float64]) -> None:
        """FIXME! briefly describe function

        :param luminosity:
        :returns:
        :rtype:

        """

        self._luminosity = luminosity  # type: NDArray[np.float64]

    def set_distance(self, distance: NDArray[np.float64]) -> None:
        """FIXME! briefly describe function

        :param distance:
        :returns:
        :rtype:

        """

        self._distance = distance  # type: NDArray[np.float64]

    def set_selection_probability(self, selector: SelectionProbabilty) -> None:

        assert isinstance(
            selector, SelectionProbabilty
        ), "The selector is not a valid selection probability"

        self._selector = selector  # type: SelectionProbabilty

    def _apply_selection(self) -> None:
        """
        Default selection if none is specfied in child class
        """

        self._selector.draw(len(self._obs_values))

        # self._selection = np.ones_like(
        #     self._obs_values, dtype=bool
        # )  # type: NDArray[np.bool_]

    def set_secondary_sampler(self, sampler) -> None:
        """
        Allows the setting of a secondary sampler from which to derive values
        """

        # make sure we set the sampler as a secondary
        # this causes it to throw a flag in the main
        # loop if we try to add it again

        sampler.make_secondary()
        # attach the sampler to this class

        self._secondary_samplers[sampler.name] = sampler
        self._has_secondary = True  # type: bool

    def draw(self, size: int = 1, verbose: bool = True):
        """
        Draw the primary and secondary samplers. This is the main call.

        :param size: the number of samples to draw
        """
        # do not resample!
        if not self._is_sampled:
            if verbose:
                print("Sampling: %s" % self.name)

            if self._has_secondary:
                if verbose:
                    print("%s is sampling its secondary quantities" % self.name)

            if self._uses_distance:
                self._selector.set_distance(self._distance)

            if self._uses_luminoity:
                self._selector.set_luminosity(self._luminosity)

            for k, v in self._secondary_samplers.items():

                assert v.is_secondary, "Tried to sample a non-secondary, this is a bag"

                # we do not allow for the secondary
                # quantities to derive a luminosity
                # as it should be the last thing dervied

                v.draw(size=size, verbose=verbose)

            # Now, it is assumed that if this sampler depends on the previous samplers,
            # then those properties have been drawn

            self.true_sampler(size=size)

            if self._is_observed:

                self.observation_sampler(size)

            else:

                self._obs_values = (
                    self._true_values
                )  # type: NDArray[(size,),np.float64]

            self._selector.set_observed_value(self._obs_values)

            # check to make sure we sampled!
            assert (
                self.true_values is not None and len(self.true_values) == size
            ), f"{self.name} likely has a bad true_sampler function"
            assert (
                self.obs_values is not None and len(self.obs_values) == size
            ), f"{self.name} likely has a observation_sampler function"

            # now apply the selection to yourself
            # if there is nothing coded, it will be
            # list of all true

            self._apply_selection()

            self._is_sampled = True

    def make_secondary(self) -> None:

        self._is_secondary = True  # type: bool

    def get_secondary_properties(
        self,
        recursive_secondaries: Union[Dict[str, NDArray[Any]], None] = None,
        graph=None,
        primary=None,
        spatial_distribution=None,
    ) -> SamplerDict:
        """FIXME! briefly describe function

        :param recursive_secondaries:
        :returns:
        :rtype:

        """

        # if a holder was not passed, create one
        if recursive_secondaries is None:

            recursive_secondaries = {}  # type: SamplerDict

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

                recursive_secondaries = v.get_secondary_properties(
                    recursive_secondaries, graph, k, spatial_distribution
                )  # type: SamplerDict

        # add our own on
        recursive_secondaries[self._name] = {
            "true_values": self._true_values,
            "obs_values": self._obs_values,
            "selection": self._selector,
        }

        return recursive_secondaries

    @property
    def secondary_samplers(self) -> SamplerDict:
        """
        Secondary samplers
        """

        return self._secondary_samplers

    @property
    def is_secondary(self) -> bool:

        return self._is_secondary

    @property
    def has_secondary(self) -> bool:

        return self._has_secondary

    @property
    def observed(self) -> bool:
        """"""
        return self._is_observed

    @property
    def name(self) -> str:
        return self._name

    @property
    def obs_name(self) -> str:
        return self._obs_name

    @property
    def true_values(self) -> NDArray[np.float64]:
        """
        The true values

        :returns:
        :rtype:

        """

        return self._true_values

    @property
    def obs_values(self) -> NDArray[np.float64]:
        """
        The observed values
        :returns:
        :rtype:

        """

        return self._obs_values

    @property
    def selection(self) -> NDArray[np.bool_]:
        """
        The selection function

        :returns:
        :rtype: np.ndarray

        """

        return self._selector.selection

    @property
    def selector(self) -> SelectionProbabilty:
        return self._selector

    @property
    def truth(self) -> Dict[str, float]:
        return self._parameter_storage

    @property
    def uses_distance(self) -> bool:
        return self._uses_distance

    @property
    def uses_luminosity(self) -> NDArray[np.float64]:
        return self._luminosity

    @abc.abstractmethod
    def true_sampler(self, size: int = 1):

        pass

    def observation_sampler(self, size: int = 1) -> NDArray[np.float64]:

        return self._true_values


class NonObservedAuxSampler(AuxiliarySampler):
    def __init__(
        self, name: str, uses_distance: bool = False, uses_luminosity: bool = False
    ):

        super(NonObservedAuxSampler, self).__init__(
            name=name,
            observed=False,
            uses_distance=uses_distance,
            uses_luminosity=uses_luminosity,
        )


class DerivedLumAuxSampler(AuxiliarySampler):
    def __init__(self, name: str, uses_distance: bool = False):
        """FIXME! briefly describe function

        :param name:
        :param sigma:
        :param observed:
        :returns:
        :rtype:

        """

        super(DerivedLumAuxSampler, self).__init__(
            name, observed=False, uses_distance=uses_distance
        )

    @abc.abstractmethod
    def compute_luminosity(self):

        raise RuntimeError("Must be implemented in derived class")
