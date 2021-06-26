from popsynth.utils.logging import setup_logger

from .generic_selectors import LowerBound, SoftSelection

log = setup_logger(__name__)


class HardFluxSelection(LowerBound):
    _selection_name = "HardFluxSelection"

    def __init__(self) -> None:
        """
        A hard selection on the observed flux.

        Based on :class:`LowerBound`.
        """

        super(HardFluxSelection, self).__init__(use_flux=True)

    def draw(self, size: int):

        super(HardFluxSelection, self).draw(size)


class SoftFluxSelection(SoftSelection):
    _selection_name = "SoftFluxSelection"

    def __init__(self) -> None:
        """
        A soft selection on the observed flux.

        Based on :class:`SoftSelection`.
        """

        super(SoftFluxSelection, self).__init__(use_flux=True)

    def draw(
        self,
        size: int,
    ):

        super(SoftFluxSelection, self).draw(size, use_log=True)


__all__ = ["HardFluxSelection", "SoftFluxSelection"]
