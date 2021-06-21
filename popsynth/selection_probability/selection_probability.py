import abc
from typing import Any, Dict, Optional

import numpy as np
from class_registry import AutoRegister

from popsynth.utils.logging import setup_logger
from popsynth.utils.meta import Parameter, ParameterMeta
from popsynth.utils.registry import selection_registry

log = setup_logger(__name__)


class SelectionParameter(Parameter):
    pass


class SelectionProbabilty(object,
                          metaclass=AutoRegister(selection_registry,
                                                 base_type=ParameterMeta)):
    def __init__(self,
                 name: str = "name",
                 use_obs_value: bool = False,
                 use_distance: bool = False,
                 use_luminosity: bool = False,
                 use_flux: bool = False) -> None:

        self._name = name  # type: str

        self._parameter_storage = {}

        self._observed_flux = None  # type: np.ndarray
        self._observed_value = None  # type: np.ndarray
        self._distance = None  # type: np.ndarray
        self._luminosity = None  # type: np.ndarray
        self._selection = None  # type: np.ndarray
        self._is_sampled: bool = False

        self._use_obs_value: bool = use_obs_value
        self._use_distance: bool = use_distance
        self._use_luminosity: bool = use_luminosity
        self._use_flux: bool = use_flux

    def __add__(self, other):

        log.debug(f"adding selection from {other.name} to {self.name}")

        new_selection = np.logical_and(self._selection, other.selection)

        out = DummySelection()
        out._selection = new_selection
        return out

    # def __radd__(self, other):

    #     return self.__add__(other)

    def set_luminosity(self, luminosity: np.ndarray) -> None:
        """
        set the luminosity of the selection

        :param luminosity:
        :returns:
        :rtype:

        """

        self._luminosity = luminosity  # type: np.ndarray

    def set_distance(self, distance: np.ndarray) -> None:
        """
        set the distance of the selection

        :param distance:
        :returns:
        :rtype:

        """

        self._distance = distance  # type: np.ndarray

    def set_observed_flux(self, observed_flux: np.ndarray) -> None:
        """
        set the observed flux of the selection

        :param luminosity:
        :returns:
        :rtype:

        """

        self._observed_flux = observed_flux  # type: np.ndarray

    def set_observed_value(self, observed_value: np.ndarray) -> None:
        self._observed_value = observed_value  # type: np.ndarray

    @abc.abstractclassmethod
    def draw(self, size: int) -> None:

        pass

    def reset(self):
        """
        resest the selector
        """
        if self._is_sampled:

            log.info(f"Selection: {self.name} is being reset")

            self._is_sampled = False
            self._selection = None

    def select(self, size: int):

        if not self._is_sampled:

            self.draw(size)

        else:

            log.warning(f"selecting with {self.name} more than once!")

        log.debug(f"{self.name} sampled {sum(self._selection)} objects")

        self._is_sampled = True

    @property
    def parameters(self) -> Dict[str, float]:

        out = {}

        for k, v in self._parameter_storage.items():

            if v is not None:

                out[k] = v

        return out

    @property
    def selection(self) -> np.ndarray:
        if self._selection is not None:

            return self._selection

        else:

            log.error(f"selector: {self.name} as not be sampled yet!")
            raise RuntimeError()

    @property
    def n_selected(self) -> int:
        return sum(self._selection)

    @property
    def n_non_selected(self) -> int:
        return sum(~self._selection)

    @property
    def n_objects(self) -> int:
        return self._selection.shape[0]

    @property
    def selection_index(self) -> np.ndarray:
        return np.where(self._selection)[0]

    @property
    def non_selection_index(self) -> np.ndarray:
        return np.where(~self._selection)[0]

    @property
    def name(self) -> str:
        return self._name


class DummySelection(SelectionProbabilty):
    _selection_name = "DummySelection"

    def __init__(self):

        super(DummySelection, self).__init__(name="dummy")

    def draw(self, size=1):
        pass
