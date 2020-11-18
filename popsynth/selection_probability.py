import abc
from typing import Any

import numpy as np
import scipy.special as sf
import scipy.stats as stats
from nptyping import NDArray
from tqdm.autonotebook import tqdm as progress_bar


class SelectionVariableContaier(object):
    def __init__(self):
        pass


class SelectionProbabilty(object, metaclass=abc.ABCMeta):
    def __init__(self, name: str = "name") -> None:

        self._name = name  # type: str

        self._observed_flux = None  # type: NDArray[np.float64]
        self._distance = None  # type: NDArray[np.float64]
        self._luminosity = None  # type: NDArray[np.float64]

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

    def set_observed_flux(self, observed_flux: NDArray[np.float64]) -> None:
        """FIXME! briefly describe function

        :param luminosity:
        :returns:
        :rtype:

        """

        self._observed_flux = observed_flux  # type: NDArray[np.float64]

    @abc.abstractclassmethod
    def draw(self, size: int, verbose: bool = False) -> None:

        pass

    @property
    def selection(self) -> NDArray[np.bool_]:
        return self._selection

    @property
    def selection_index(self) -> NDArray[np.int64]:
        return np.where(self._selection)[0]

    @property
    def non_selection_index(self) -> NDArray[np.int64]:
        return np.where(~self._selection)[0]

    @property
    def name(self) -> str:
        return self._name


class UnitySelection(SelectionProbabilty):
    def __init__(self):
        """
        A selection that returns all unity
        """
        super(UnitySelection, self).__init__()

    def draw(self, size: int) -> None:

        self._selection = np.ones(size, dtype=bool)


class BernoulliSelection(SelectionProbabilty):
    def __init__(self, probabilty: float = 0.5) -> None:

        assert probabilty <= 1.0
        assert probabilty >= 0.0

        super(BernoulliSelection, self).__init__()

        self._probabilty = probabilty  # type: float

    def draw(self, size: int, verbose: bool = False) -> None:

        if verbose:

            self._selection = np.zeros(size)  # type: NDArray[(size,), bool]

            with progress_bar(size, desc=f"Selecting {self._name}") as pbar2:
                for i in range(size):

                    # see if we detect the distance
                    if stats.bernoulli.rvs(self._probabailty) == 1:

                        self._selection[i] = 1

                    pbar2.update()

        else:

            self._selection = stats.bernoulli.rvs(self._probabailty, size=size).astype(
                bool
            )  # type: NDArray[(size,), bool]


class HardSelection(SelectionProbabilty):
    def __init__(self, boundary: float):

        super(HardSelection, self).__init__(name="Hard selection")

        self._boundary = boundary  # type: float

    def _draw(self, values) -> NDArray[np.bool_]:

        return values >= self._boundary


class HardFluxSelection(HardSelection):
    def __init__(self, boundary: float) -> None:

        assert boundary >= 0

        super(HardFluxSelection, self).__init__(self, boundary)

    def draw(self, size: int, verbose: bool = False):

        self._selection = self._draw(self._observed_flux)  # type: NDArray[np.bool]


class SoftSelection(SelectionProbabilty):
    def __init__(self, boundary: float, strength: float) -> None:

        self._strength = strength  # type: float
        self._boundary = boundary  # type: float

        super(SoftSelection, self).__init__(name="Soft Selection")

    def _draw(self, size: int, values: NDArray[np.float64]) -> NDArray[np.bool_]:

        probs = sf.expit(
            self._strength * (values - self._boundary)
        )  # type: NDArray[np.float64]

        return stats.bernoulli.rvs(probs, size=size)


class SoftFluxSelection(SoftSelection):
    def __init__(self, boundary: float, strength: float) -> None:

        super(SoftFluxSelection, self).__init__(np.log10(boundary), strength)

    def draw(self, size: int, verbose: bool = False):

        self._selection = self._draw(size, np.log10(self._observed_flux))
