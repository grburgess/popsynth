import numpy as np

from popsynth.utils.logging import setup_logger

from .generic_selectors import LowerBound, SoftSelection

log = setup_logger(__name__)


class HardFluxSelection(LowerBound):
    _selection_name = "HardFluxSelection"

    def __init__(self) -> None:

        super(HardFluxSelection, self).__init__()

    def draw(self, size: int):

        super(HardFluxSelection, self).draw(self._observed_flux)


class SoftFluxSelection(SoftSelection):
    _selection_name = "SoftFluxSelection"

    def __init__(self) -> None:

        super(SoftFluxSelection, self).__init__()

    def draw(
        self,
        size: int,
    ):

        super(SoftFluxSelection, self).draw(self._observed_flux, use_log=True)


__all__ = ["HardFluxSelection", "SoftFluxSelection"]
