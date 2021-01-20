from popsynth.distribution import SpatialDistribution

from .selection_probability import SelectionProbabilty


class SpatialSelection(SelectionProbabilty):
    def __init__(self, name) -> None:

        super(SpatialSelection, self).__init__(name)

        self._spatial_distribution: SpatialDistribution = None

    def set_spatial_distribution(
            self, spatial_distribtuion: SpatialDistribution) -> None:
        """
        set the spatial distribution
        """
        assert isinstance(spatial_distribtuion, SpatialDistribution)

        self._spatial_distribution: SpatialDistribution = spatial_distribtuion
