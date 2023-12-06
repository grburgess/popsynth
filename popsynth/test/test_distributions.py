import numpy as np

from popsynth.distribution import LuminosityDistribution, SpatialDistribution
from popsynth.distributions.delta_distribution import DeltaDistribution
from popsynth.distributions.flatland_distribution import FlatlandDistribution
from popsynth.distributions.spiral_galaxy_distribution import (
    SpiralGalaxyDistribution, )
from popsynth.population_synth import PopulationSynth


class DummySDistribution(SpatialDistribution):

    def __init__(self, seed=1234, form=None, truth={}):

        # the latex formula for the ditribution
        form = r"4 \pi r2"

        # we do not need a "truth" dict here because
        # there are no parameters

        super(DummySDistribution, self).__init__(seed=seed,
                                                 name="dummy1",
                                                 form=form)

    def differential_volume(self, r):

        # the differential volume of a sphere
        return 4 * np.pi * r * r

    def transform(self, L, r):

        # luminosity to flux
        return L / (4.0 * np.pi * r * r)

    def dNdV(self, r):
        return 1.0


class DummyLDistribution(LuminosityDistribution):

    def __init__(self, seed=1234, name="dummy"):

        # the latex formula for the ditribution
        lf_form = r"1"

        super(DummyLDistribution, self).__init__(seed=seed,
                                                 name="pareto",
                                                 form=lf_form)

    def phi(self, L):

        # the actual function, only for plotting

        return 0

    def draw_luminosity(self, size=1):
        # how to sample the latent parameters
        return np.random.uniform(0, 1, size=size)


class MyPopulation(PopulationSynth):

    def __init__(self, r_max=5, seed=1234):

        # instantiate the distributions
        luminosity_distribution = DummyLDistribution(seed=seed)
        spatial_distribution = DummySDistribution()
        spatial_distribution.r_max = r_max

        # pass to the super class
        super(MyPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )


class MyFlatPopulation(PopulationSynth):

    def __init__(self, r_max=5, seed=1234):

        # instantiate the distributions
        luminosity_distribution = DummyLDistribution(seed=seed)
        spatial_distribution = FlatlandDistribution()
        spatial_distribution.Lambda = 1
        spatial_distribution.r_max = r_max

        # pass to the super class
        super(MyFlatPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )


class MyDeltaPopulation(PopulationSynth):

    def __init__(self, r_max=5, seed=1234):

        # instantiate the distributions
        luminosity_distribution = DeltaDistribution()
        luminosity_distribution.Lp = 1

        spatial_distribution = FlatlandDistribution()
        spatial_distribution.Lambda = 1
        spatial_distribution.r_max = r_max

        super(MyDeltaPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )


def test_distribution_with_no_parameters():

    popgen = MyPopulation()

    popgen.draw_survey()


def test_flatland():

    popgen = MyFlatPopulation()

    popgen.draw_survey()


def test_spiral():

    import popsynth

    popsynth.update_logging_level("INFO")
    from popsynth.populations.spatial_populations import MWRadialPopulation

    ld = popsynth.distributions.pareto_distribution.ParetoDistribution()
    ld.alpha = 3
    ld.Lmin = 1
    synth = MWRadialPopulation(rho=1, luminosity_distribution=ld)
    population = synth.draw_survey()


def test_delta():

    popgen = MyDeltaPopulation()

    popgen.draw_survey()


def test_delta_lf():

    luminosity_distribution = DeltaDistribution()
    luminosity_distribution.Lp = 1

    assert luminosity_distribution.phi(1) == 1
    assert luminosity_distribution.phi(2) == 0

    L = [1, 2, 3, 4]

    assert luminosity_distribution.phi(L)[0] == 1
    assert luminosity_distribution.phi(L)[1] == 0
