from popsynth.distributions.spherical_distribution import (
    ConstantSphericalDistribution,
    ZPowerSphericalDistribution,
)
from popsynth.distributions.cosmological_distribution import (
    SFRDistribtution,
    MergerDistribution,
    
)

from popsynth.population_synth import PopulationSynth

## First create a bunch of spatial distributions


class SphericalPopulation(PopulationSynth):
    def __init__(self, Lambda, r_max=10.0, seed=1234, luminosity_distribution=None):

        spatial_distribution = ConstantSphericalDistribution(
            Lambda=Lambda, r_max=r_max, seed=seed
        )

        super(SphericalPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )


class ZPowerSphericalPopulation(PopulationSynth):
    def __init__(
        self, Lambda, delta, r_max=10.0, seed=1234, luminosity_distribution=None
    ):

        spatial_distribution = ZPowerConstantSphericalDistribution(
            Lambda=Lambda, delta=delta, r_max=r_max, seed=seed
        )

        super(ZPowerSphericalPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )


class SFRPopulation(PopulationSynth):
    def __init__(
        self, r0, rise, decay, peak, r_max=10, seed=1234, luminosity_distribution=None
    ):

        spatial_distribution = SFRDistribtution(
            r0=r0, rise=rise, decay=decay, peak=peak, r_max=r_max, seed=seed
        )

        super(SFRPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )

