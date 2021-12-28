from astropy.coordinates import SkyCoord

from popsynth.distribution import SpatialDistribution
from .selection_probability import SelectionProbability, SelectionParameter


class SpatialSelection(SelectionProbability):

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
            self, spatial_distribtuion: SpatialDistribution) -> None:
        """
        Set the spatial distribution for the selection.

        :param spatial_distribution: The spatial_distribution
        :type spatial_distribution: :class:`SpatialDistribution`
        """
        assert isinstance(spatial_distribtuion, SpatialDistribution)

        self._spatial_distribution: SpatialDistribution = spatial_distribtuion


class GalacticPlaneSelection(SpatialSelection):

    _selection_name = "GalacticPlaneSelection"

    b_limit = SelectionParameter(vmin=0, vmax=90)

    def __init__(self, name: str = "galactic plane selector"):
        """
        A selection that excludes objects near the galactic plane.

        :param name: Name of the selection
        :type name: str
        :param b_limit: Limit around Galactic plane to exclude in
            Galactic latitude and in units of degrees
        :type b_limit: :class:`SelectionParameter`
        """
        super(GalacticPlaneSelection, self).__init__(name=name)

    def draw(self, size: int):

        c = SkyCoord(
            self._spatial_distribution.ra,
            self._spatial_distribution.dec,
            unit="deg",
            frame="icrs",
        )

        b = c.galactic.b.deg

        self._selection = (b >= self.b_limit) | (b <= -self.b_limit)


__all__ = ["SpatialSelection", "GalacticPlaneSelection"]


class DistanceSelection(SpatialSelection):

    _selection_name = "DistanceSelection"

    min_distance = SelectionParameter(vmin=0)
    max_distance = SelectionParameter(vmin=0)

    def __init__(self, name: str = "distance"):
        """
        Select distances

        :param name: Name of the selection
        :type name: str
        :param min_distance: minimum distance to select
        :param max_distance: maximum distance to select
        """
        super(DistanceSelection, self).__init__(name=name)

    def draw(self, size: int):

        self._selection = (
            self._spatial_distribution.distances >= self.min_distance) & (
                self._spatial_distribution.distances <= self.max_distance)
