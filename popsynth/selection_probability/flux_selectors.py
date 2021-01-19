import numpy as np

from .generic_selectors import HardSelection, SoftSelection


class HardFluxSelection(HardSelection):
    def __init__(self, boundary: float) -> None:

        assert boundary >= 0

        super(HardFluxSelection, self).__init__(boundary)

    def draw(self, size: int, verbose: bool = False):

        self._selection = self._draw(self._observed_flux)  # type: np.ndarray


class SoftFluxSelection(SoftSelection):
    def __init__(self, boundary: float, strength: float) -> None:

        super(SoftFluxSelection, self).__init__(boundary, strength)

    def draw(self, size: int, verbose: bool = False):

        self._selection = self._draw(size, self._observed_flux, use_log=True)
