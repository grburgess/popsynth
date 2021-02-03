import numpy as np
import scipy.stats as stats
import scipy.special as sf
from popsynth.utils.progress_bar import progress_bar

from .selection_probability import SelectionProbabilty

from popsynth.utils.logging import setup_logger
from popsynth.utils.configuration import popsynth_config

log = setup_logger(__name__)


class UnitySelection(SelectionProbabilty):
    def __init__(self):
        """
        A selection that returns all unity
        """
        super(UnitySelection, self).__init__(name="unity")

    def draw(self, size: int) -> None:

        self._selection = np.ones(size, dtype=int).astype(bool)


class BernoulliSelection(SelectionProbabilty):
    def __init__(self, probability: float = 0.5) -> None:

        assert probability <= 1.0
        assert probability >= 0.0

        super(BernoulliSelection, self).__init__(name="Bernoulli")

        self._probability = probability  # type: float

    def draw(self, size: int) -> None:

        if popsynth_config["show_progress"]:

            self._selection = np.zeros(size, dtype=int).astype(
                bool)  # type: np.ndarray

            for i in progress_bar(range(size), desc=f"Selecting {self.name}"):

                # see if we detect the distance
                if stats.bernoulli.rvs(self._probability) == 1:

                    self._selection[i] = 1

        else:

            self._selection = stats.bernoulli.rvs(
                self._probability, size=size).astype(bool)  # type: np.ndarray

    @property
    def probability(self) -> float:
        return self._probability


class HardSelection(SelectionProbabilty):
    def __init__(self, boundary: float):

        super(HardSelection, self).__init__(name="Hard selection")

        self._boundary = boundary  # type: float

    def _draw(self, values) -> np.ndarray:

        return values >= self._boundary

    @property
    def boundary(self):
        return self._boundary

    @property
    def hard_cut(self):
        return True


class SoftSelection(SelectionProbabilty):
    def __init__(self, boundary: float, strength: float) -> None:

        self._strength = strength  # type: float
        self._boundary = boundary  # type: float

        super(SoftSelection, self).__init__(name="Soft Selection")

    def _draw(self,
              size: int,
              values: np.ndarray,
              use_log=False) -> np.ndarray:

        if not use_log:
            probs = sf.expit(self._strength *
                             (values - self._boundary))  # type: np.ndarray

        else:

            probs = sf.expit(self._strength *
                             (np.log10(values) -
                              np.log10(self._boundary)))  # type: np.ndarray

        return stats.bernoulli.rvs(probs, size=size).astype(bool)

    @property
    def boundary(self):
        return self._boundary

    @property
    def strength(self):
        return self._strength

    @property
    def hard_cut(self):
        return False
