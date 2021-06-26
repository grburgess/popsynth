from popsynth.distribution import SpatialDistribution

from .selection_probability import SelectionProbabilty


class SpatialSelection(SelectionProbabilty):

    _selection_name = "SpatialSelection"

    def __init__(self, name: str) -> None:
        """
        A generic spatial selection.

        :param name: Name of the selection
        :type name: str
        """

        super(SpatialSelection, self).__init__(name)

        self._spatial_distribution: SpatialDistribution = None

    def set_spatial_distribution(
        self, spatial_distribtuion: SpatialDistribution
    ) -> None:
        """
        Set the spatial distribution for the selection.

        :param spatial_distribution: The spatial_distribution
        :type spatial_distribution: :class:`SpatialDistribution`
        """
        assert isinstance(spatial_distribtuion, SpatialDistribution)

        self._spatial_distribution: SpatialDistribution = spatial_distribtuion


__all__ = ["SpatialSelection"]
