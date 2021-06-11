import numpy as np
from astropy.coordinates import SkyCoord

import popsynth


class GalacticPlaceSelection(popsynth.SpatialSelection):

    _selection_name = "GalacticPlaceSelection"

    b_limit = popsynth.SelectionParameter(vmin=0, vmax=90)

    def __init__(self, name="mw plane selector"):
        """
        places a limit above the galactic plane for objects
        """
        super(GalacticPlaceSelection, self).__init__(name=name)

    def draw(self, size: int):

        g_coor = SkyCoord(self._spatial_distribution.ra, self._spatial_distribution.dec, unit="deg",
                          frame="icrs").transform_to("galactic")

        self._selection = (g_coor.b.deg >= self.b_limit) | (
            g_coor.b.deg <= -self.b_limit)


def test_spatial_selection():

    pg = popsynth.populations.Log10NormalHomogeneousSphericalPopulation(
        10, 1, 1)

    gps = GalacticPlaceSelection()
    gps.b_limit = 10.

    pg.add_spatial_selector(gps)

    pg.draw_survey()

    d = pg.to_dict()

    ps = popsynth.PopulationSynth.from_dict(d)
