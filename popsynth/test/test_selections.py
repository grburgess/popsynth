import popsynth
from popsynth.selection_probability.spatial_selection import (
    GalacticPlaneSelection, )


def test_spatial_selection():

    pg = popsynth.populations.Log10NormalHomogeneousSphericalPopulation(
        10, 1, 1)

    gps = GalacticPlaneSelection()
    gps.b_limit = 10.0

    pg.add_spatial_selector(gps)

    pg.draw_survey()

    d = pg.to_dict()

    ps = popsynth.PopulationSynth.from_dict(d)
