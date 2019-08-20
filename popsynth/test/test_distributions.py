import numpy as np

from popsynth.distribution import SpatialDistribution
from popsynth.distribution import LuminosityDistribution
from popsynth.population_synth import PopulationSynth


class DummySDistribution(SpatialDistribution):
    def __init__(self, r_max=10, seed=1234, form=None, truth={}):

        # the latex formula for the ditribution
        form = r"4 \pi r2"

        # we do not need a "truth" dict here because
        # there are no parameters

        super(DummySDistribution, self).__init__(
            r_max=r_max, seed=seed, name="dummy1", form=form, truth={}
        )

    def differential_volume(self, r):

        # the differential volume of a sphere
        return 4 * np.pi * r * r

    def transform(self, L, r):

        # luminosity to flux
        return L / (4.0 * np.pi * r * r)


class DummyLDistribution(LuminosityDistribution):
    def __init__(self, seed=1234, name="dummy"):


        # the latex formula for the ditribution
        lf_form = r"1"

        super(DummyLDistribution, self).__init__(
            seed=seed, name="pareto", form=lf_form
        )

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
        spatial_distribution = DummySDistribution(r_max=r_max)

        # pass to the super class
        super(MyPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )


def test_distribution_with_no_parameters():

    popgen = MyPopulation()

    popgen.draw_survey(1E-50)
