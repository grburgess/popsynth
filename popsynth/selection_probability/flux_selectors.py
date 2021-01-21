import numpy as np

from popsynth.utils.logging import setup_logger

from .generic_selectors import HardSelection, SoftSelection

log = setup_logger(__name__)


class HardFluxSelection(HardSelection):
    def __init__(self, boundary: float) -> None:

        assert boundary >= 0

        log.debug(f"created a hard flux selection with boundary {boundary}")

        super(HardFluxSelection, self).__init__(boundary)

    def draw(self, size: int):

        self._selection = self._draw(self._observed_flux)  # type: np.ndarray


class SoftFluxSelection(SoftSelection):
    def __init__(self, boundary: float, strength: float) -> None:

        log.debug(
            f"created a hard flux selection with boundary {boundary} and strenhth {strength}")

        super(SoftFluxSelection, self).__init__(boundary, strength)

    def draw(self, size: int,):

        self._selection = self._draw(size, self._observed_flux, use_log=True)
