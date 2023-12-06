import numpy as np
import scipy.special as sf
import scipy.stats as stats

from popsynth.utils.configuration import popsynth_config
from popsynth.utils.logging import setup_logger
from popsynth.utils.progress_bar import progress_bar

from .selection_probability import SelectionParameter, SelectionProbability

log = setup_logger(__name__)


class UnitySelection(SelectionProbability):
    _selection_name = "UnitySelection"

    def __init__(self, name="unity"):
        """
        A selection that returns all selected.

        :param name: Name of the selection
        """
        super(UnitySelection, self).__init__(name=name)

    def draw(self, size: int) -> None:

        self._selection = np.ones(size, dtype=int).astype(bool)


class BernoulliSelection(SelectionProbability):
    _selection_name = "BernoulliSelection"

    probability = SelectionParameter(vmin=0, vmax=1, default=0.5)

    def __init__(self) -> None:
        """
        A Bernoulli selection with ``probability`` as a parameter.

        :param probability: Probability for each Bernoulli trial
        :type probability: :class:`SelectionParameter`
        """

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
                self.probability, size=size).astype(bool)  # type: np.ndarray


class BoxSelection(SelectionProbability):
    _selection_name = "BoxSelection"

    vmin = SelectionParameter()
    vmax = SelectionParameter()

    def __init__(
        self,
        name: str = "box selection",
        use_obs_value: bool = False,
        use_distance: bool = False,
        use_luminosity: bool = False,
        use_flux: bool = False,
    ):
        """
        A box selection on observed_value, distance,
        luminosity or flux.

        :param name: Name of the selection
        :type name: str
        :param use_obs_value: If `True`, make selection on
            observed_value. `False` by default.
        :type use_obs_value: bool
        :param use_distance: If `True` make selection on distance.
            `False` by default.
        :type use_distance: bool
        :param use_luminosity: If `True` make selection on luminosity.
            `False` by default.
        :type use_luminosity: bool
        :param use_flux: If `True` make selection on flux. `False` by default.
        :type use_flux: bool
        :param vmin: Minimum value of selection
        :type vmin: :class:`SelectionParameter`
        :param vmax: Maximum value of selection
        :type vmax: :class:`SelectionParameter`
        """

        super(BoxSelection, self).__init__(
            name=name,
            use_distance=use_distance,
            use_luminosity=use_luminosity,
            use_obs_value=use_obs_value,
            use_flux=use_flux,
        )

    def draw(self, size: int) -> np.ndarray:

        if self._use_distance:
            values = self._distance

        if self._use_obs_value:

            values = self._observed_value

        if self._use_luminosity:

            values = self._luminosity

        if self._use_flux:

            values = self._observed_flux

        self._selection = (values >= self.vmin) & (values <= self.vmax)


class LowerBound(SelectionProbability):
    _selection_name = "LowerBound"

    boundary = SelectionParameter()

    def __init__(
        self,
        name: str = "Hard selection",
        use_obs_value: bool = False,
        use_distance: bool = False,
        use_luminosity: bool = False,
        use_flux: bool = False,
    ):
        """
        A hard, lower bound selection on obs_value, distance,
        luminosity or flux.

        Selects values >= ``boundary``.

        :param name: Name of the selection
        :type name: str
        :param use_obs_value: If `True`, make selection on
            observed_value. `False` by default.
        :type use_obs_value: bool
        :param use_distance: If `True` make selection on distance.
            `False` by default.
        :type use_distance: bool
        :param use_luminosity: If `True` make selection on luminosity.
            `False` by default.
        :type use_luminosity: bool
        :param use_flux: If `True` make selection on flux. `False` by default.
        :type use_flux: bool
        :param boundary: Value of the selection boundary
        :type boundary: :class:`SelectionParameter`
        """
        super(LowerBound, self).__init__(
            name=name,
            use_distance=use_distance,
            use_luminosity=use_luminosity,
            use_obs_value=use_obs_value,
            use_flux=use_flux,
        )

    def draw(self, size: int) -> None:

        if self._use_distance:
            values = self._distance

        if self._use_obs_value:

            values = self._observed_value

        if self._use_luminosity:

            values = self._luminosity

        if self._use_flux:

            values = self._observed_flux

        self._selection = values >= self.boundary


class UpperBound(SelectionProbability):
    _selection_name = "UpperBound"

    boundary = SelectionParameter()

    def __init__(
        self,
        name: str = "Hard selection",
        use_obs_value: bool = False,
        use_distance: bool = False,
        use_luminosity: bool = False,
        use_flux: bool = False,
    ):
        """
        A hard, upper bound selection on obs_value, distance,
        luminosity or flux.

        Selects values <= ``boundary``.

        :param name: Name of the selection
        :type name: str
        :param use_obs_value: If `True`, make selection on
            observed_value. `False` by default.
        :type use_obs_value: bool
        :param use_distance: If `True` make selection on distance.
            `False` by default.
        :type use_distance: bool
        :param use_luminosity: If `True` make selection on luminosity.
            `False` by default.
        :type use_luminosity: bool
        :param use_flux: If `True` make selection on flux. `False` by default.
        :type use_flux: bool
        :param boundary: Value of the selection boundary
        :type boundary: :class:`SelectionParameter`
        """
        super(UpperBound, self).__init__(
            name=name,
            use_distance=use_distance,
            use_luminosity=use_luminosity,
            use_obs_value=use_obs_value,
            use_flux=use_flux,
        )

    def draw(self, size: int) -> None:

        if self._use_distance:
            values = self._distance

        if self._use_obs_value:

            values = self._observed_value

        if self._use_luminosity:

            values = self._luminosity

        if self._use_flux:

            values = self._observed_flux

        self._selection = values <= self.boundary


class SoftSelection(SelectionProbability):
    _selection_name = "SoftSelection"

    boundary = SelectionParameter()
    strength = SelectionParameter(vmin=0)

    def __init__(
        self,
        name: str = "Soft Selection",
        use_obs_value: bool = False,
        use_distance: bool = False,
        use_luminosity: bool = False,
        use_flux: bool = False,
    ) -> None:
        """
        A soft selection using an inverse logit function either on the
        log or linear value of the observed_value, distance,
        luminosity or flux.

        :param name: Name of the selection
        :type name: str
        :param use_obs_value: If `True`, make selection on
            observed_value. `False` by default.
        :type use_obs_value: bool
        :param use_distance: If `True` make selection on distance.
            `False` by default.
        :type use_distance: bool
        :param use_luminosity: If `True` make selection on luminosity.
            `False` by default.
        :type use_luminosity: bool
        :param use_flux: If `True` make selection on flux. `False` by default.
        :type use_flux: bool
        :param boundary: Center of the inverse logit
        :type boundary: :class:`SelectionParameter`
        :param strength: Width of the logit
        :type strength: :class:`SelectionParameter`
        """

        super(SoftSelection, self).__init__(
            name=name,
            use_distance=use_distance,
            use_luminosity=use_luminosity,
            use_obs_value=use_obs_value,
            use_flux=use_flux,
        )

    def draw(self, size: int, use_log=True) -> None:

        if self._use_distance:
            values = self._distance

        if self._use_obs_value:

            values = self._observed_value

        if self._use_luminosity:

            values = self._luminosity

        if self._use_flux:

            values = self._observed_flux

        if not use_log:
            probs = sf.expit(self.strength *
                             (values - self.boundary))  # type: np.ndarray

        else:

            probs = sf.expit(self.strength *
                             (np.log10(values) -
                              np.log10(self.boundary)))  # type: np.ndarray

        self._selection = stats.bernoulli.rvs(probs,
                                              size=len(values)).astype(bool)

    __all__ = [
        "UnitySelection",
        "BernoulliSelection",
        "BoxSelection",
        "UpperBound",
        "UpperBound",
        "SoftSelection",
    ]
