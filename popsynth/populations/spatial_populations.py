from popsynth.distributions.spherical_distribution import (
    ConstantSphericalDistribution,
    ZPowerSphericalDistribution,
)
from popsynth.distributions.cosmological_distribution import (
    SFRDistribution,
    # MergerDistribution,
    ZPowerCosmoDistribution,
)

from popsynth.distributions.spiral_galaxy_distribution import SpiralGalaxyDistribution

from popsynth.population_synth import PopulationSynth

## First create a bunch of spatial distributions


class SphericalPopulation(PopulationSynth):
    def __init__(self,
                 Lambda,
                 r_max=5.0,
                 seed=1234,
                 luminosity_distribution=None):
        """FIXME! briefly describe function

        :param Lambda:
        :param r_max:
        :param seed:
        :param luminosity_distribution:
        :returns:
        :rtype:

        """

        spatial_distribution = ConstantSphericalDistribution(seed=seed)
        spatial_distribution.Lambda = Lambda
        spatial_distribution.r_max = r_max

        super(SphericalPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )


class ZPowerSphericalPopulation(PopulationSynth):
    def __init__(self,
                 Lambda,
                 delta,
                 r_max=5.0,
                 seed=1234,
                 luminosity_distribution=None):
        """FIXME! briefly describe function

        :param Lambda:
        :param delta:
        :param r_max:
        :param seed:
        :param luminosity_distribution:
        :returns:
        :rtype:

        """

        spatial_distribution = ZPowerSphericalDistribution(seed=seed)

        spatial_distribution.Lambda = Lambda
        spatial_distribution.r_max = r_max
        spatial_distribution.delta = delta

        super(ZPowerSphericalPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )


class ZPowerCosmoPopulation(PopulationSynth):
    def __init__(
        self,
        Lambda,
        delta,
        r_max=5.0,
        seed=1234,
        luminosity_distribution=None,
        is_rate=True,
    ):
        """FIXME! briefly describe function

        :param Lambda:
        :param delta:
        :param r_max:
        :param seed:
        :param luminosity_distribution:
        :returns:
        :rtype:

        """

        spatial_distribution = ZPowerCosmoDistribution(seed=seed,
                                                       is_rate=is_rate)

        spatial_distribution.Lambda = Lambda
        spatial_distribution.r_max = r_max
        spatial_distribution.delta = delta

        super(ZPowerCosmoPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )


class SFRPopulation(PopulationSynth):
    def __init__(
        self,
        r0,
        rise,
        decay,
        peak,
        r_max=5,
        seed=1234,
        luminosity_distribution=None,
        is_rate=True,
    ):
        """FIXME! briefly describe function

        :param r0:
        :param rise:
        :param decay:
        :param peak:
        :param r_max:
        :param seed:
        :param luminosity_distribution:
        :returns:
        :rtype:

        """

        spatial_distribution = SFRDistribution(seed=seed, is_rate=is_rate)

        spatial_distribution.r0 = r0
        spatial_distribution.rise = rise
        spatial_distribution.decay = decay
        spatial_distribution.peak = peak
        spatial_distribution.r_max = r_max

        super(SFRPopulation, self).__init__(
            spatial_distribution=spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )


class MWRadialPopulation(PopulationSynth):
    def __init__(
        self,
        rho,
        a=1.64,
        b=4.01,
        R1=0.55,
        R0=8.5,
        r_max=20,
        seed=1234,
        luminosity_distribution=None,
    ):

        spatial_distribution = SpiralGalaxyDistribution(seed=seed)

        spatial_distribution.rho = rho
        spatial_distribution.a = a
        spatial_distribution.b = b
        spatial_distribution.R1 = R1
        spatial_distribution.R0 = R0
        spatial_distribution.r_max = r_max

        super(MWRadialPopulation, self).__init__(
            spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )
