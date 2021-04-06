import numpy as np

from popsynth.utils.logging import setup_logger

from .generic_selectors import HardSelection, SoftSelection

log = setup_logger(__name__)


class HardFluxSelection(HardSelection):
    _selection_name = "HardFluxSelection"

    def __init__(self, boundary: float) -> None:

        assert boundary >= 0

        log.debug(f"created a hard flux selection with boundary {boundary}")

        super(HardFluxSelection, self).__init__(boundary)

    def draw(self, size: int):

        super(HardFluxSelection, self).draw(self._observed_flux)


class SoftFluxSelection(SoftSelection):
    _selection_name = "SoftFluxSelection"

    def __init__(self, boundary: float, strength: float) -> None:

        log.debug(
            f"created a hard flux selection with boundary {boundary} and strenhth {strength}"
        )

        super(SoftFluxSelection, self).__init__(boundary, strength)

    def draw(
        self,
        size: int,
    ):

        super(SoftFluxSelection, self).draw(self._observed_flux, use_log=True)


__all__ = ["HardFluxSelection", "SoftFluxSelection"]
