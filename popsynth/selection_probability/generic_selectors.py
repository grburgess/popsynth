import numpy as np
import scipy.special as sf
import scipy.stats as stats

from popsynth.utils.configuration import popsynth_config
from popsynth.utils.logging import setup_logger
from popsynth.utils.progress_bar import progress_bar

from .selection_probability import SelectionParameter, SelectionProbabilty

log = setup_logger(__name__)


class UnitySelection(SelectionProbabilty):
    _selection_name = "UnitySelection"

    def __init__(self, name="unity"):
        """
        A selection that returns all unity
        """
        super(UnitySelection, self).__init__(name=name)

    def draw(self, size: int) -> None:

        self._selection = np.ones(size, dtype=int).astype(bool)


class BernoulliSelection(SelectionProbabilty):
    _selection_name = "BernoulliSelection"

    probability = SelectionParameter(vmin=0, vmax=1, default=0.5)

    def __init__(self) -> None:

        super(BernoulliSelection, self).__init__(name="Bernoulli")

    def draw(self, size: int) -> None:

        if popsynth_config.show_progress:

            self._selection = np.zeros(size, dtype=int).astype(
                bool)  # type: np.ndarray

            for i in progress_bar(range(size), desc=f"Selecting {self.name}"):

                # see if we detect the distance
                if stats.bernoulli.rvs(self.probability) == 1:

                    self._selection[i] = 1

        else:

            self._selection = stats.bernoulli.rvs(
                self._probability, size=size).astype(bool)  # type: np.ndarray


class BoxSelection(SelectionProbabilty):
    _selection_name = "BoxSelection"

    vmin = SelectionParameter()
    vmax = SelectionParameter()

    def __init__(self, name: str = "box selection"):

        super(BoxSelection, self).__init__(name=name)

    def draw(self, values) -> np.ndarray:

        self._selection = (values >= self.vmin) & (values <= self.vmax)

    @property
    def hard_cut(self):
        return True


class LowerBound(SelectionProbabilty):
    _selection_name = "LowerBound"

    boundary = SelectionParameter()

    def __init__(self, name="Hard selection"):
        """
        hard selection above the boundary
        """
        super(LowerBound, self).__init__(name=name
                                         )

    def draw(self, values) -> np.ndarray:

        self._selection = values >= self.boundary


class UpperBound(SelectionProbabilty):
    _selection_name = "UpperBound"

    boundary = SelectionParameter()

    def __init__(self, name="Hard selection"):
        """
        hard selection above the boundary
        """
        super(UpperBound, self).__init__(name=name
                                         )

    def draw(self, values) -> np.ndarray:

        self._selection = values <= self.boundary


class SoftSelection(SelectionProbabilty):
    _selection_name = "SoftSelection"

    boundary = SelectionParameter()
    strength = SelectionParameter(vmin=0)

    def __init__(self) -> None:
        """
        selection using an inverse logit function either on the 
        log are linear value of the parameter

        :param boundary: center of the logit
        :param strength: width of the logit
        """

        super(SoftSelection, self).__init__(name="Soft Selection")

    def draw(self,

             values: np.ndarray,
             use_log=False) -> np.ndarray:

        if not use_log:
            probs = sf.expit(self.strength *
                             (values - self.boundary))  # type: np.ndarray

        else:

            probs = sf.expit(self.strength *
                             (np.log10(values) -
                              np.log10(self.boundary)))  # type: np.ndarray

        self._selection = stats.bernoulli.rvs(
            probs, size=len(values)).astype(bool)

    @property
    def hard_cut(self):
        return False

    __all__ = ["UnitySelection", "BernoulliSelection",
               "BoxSelection", "UpperBound", "LowerBound", "SoftSelection"]
