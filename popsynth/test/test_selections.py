import numpy as np
from astropy.coordinates import SkyCoord

import popsynth


class GalacticPlaneSelection(popsynth.SpatialSelection):

    _selection_name = "GalacticPlaneSelection"

    b_limit = popsynth.SelectionParameter(vmin=0, vmax=90)

    def __init__(self, name="mw plane selector"):
        """
        places a limit above the galactic plane for objects
        """
        super(GalacticPlaneSelection, self).__init__(name=name)

    def draw(self, size: int):

        b = []

        for ra, dec in zip(
            self._spatial_distribution.ra, self._spatial_distribution.dec
        ):

            g_coor = SkyCoord(ra, dec, unit="deg", frame="icrs").transform_to(
                "galactic"
            )

            b.append(g_coor.b.deg)

        b = np.array(b)

        self._selection = (b >= self.b_limit) | (b <= -self.b_limit)


def test_spatial_selection():

    pg = popsynth.populations.Log10NormalHomogeneousSphericalPopulation(10, 1, 1)

    gps = GalacticPlaneSelection()
    gps.b_limit = 10.0

    pg.add_spatial_selector(gps)

    pg.draw_survey()

    d = pg.to_dict()

    ps = popsynth.PopulationSynth.from_dict(d)
