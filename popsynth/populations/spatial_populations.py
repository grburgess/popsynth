from popsynth.distributions.cosmological_distribution import (  # MergerDistribution,
    SFRDistribution, ZPowerCosmoDistribution,
)
from popsynth.distributions.spherical_distribution import (
    ConstantSphericalDistribution,
    ZPowerSphericalDistribution,
)
from popsynth.distribution import LuminosityDistribution
from popsynth.distributions.spiral_galaxy_distribution import SpiralGalaxyDistribution
from popsynth.population_synth import PopulationSynth
"""
Create a range of spatial populations that can
be expanded with luminosity distributions.
"""


class SphericalPopulation(PopulationSynth):

    def __init__(
        self,
        Lambda: float,
        r_max: float = 5.0,
        seed: int = 1234,
        luminosity_distribution: LuminosityDistribution = None,
    ):
        """
        A generic spherical population based on
        :class:`ConstantSphericalDistribution`.

        :param Lambda: Density per unit volume
        :type Lambda: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        :param luminosity_distribution: Luminosity distribution
        :type luminosity_distribution: :class:`LuminosityDistribution`, optional
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

    def __init__(
        self,
        Lambda: float,
        delta: float,
        r_max: float = 5.0,
        seed: int = 1234,
        luminosity_distribution: LuminosityDistribution = None,
    ):
        """
        A spherical population with a density that
        scales as (r+1)^delta.
        Based on :class:`ZpowerSphericalDistribution`.

        :param Lambda: Local density per unit volume
        :type Lambda: float
        :param delta: Index of spatial distribution
        :type delta: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        :param luminosity_distribution: Luminosity distribution
        :type luminosity_distribution: :class:`LuminosityDistribution`, optional
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
        Lambda: float,
        delta: float,
        r_max: float = 5.0,
        seed: int = 1234,
        luminosity_distribution: LuminosityDistribution = None,
        is_rate: bool = True,
    ):
        """
        A cosmological population with a density that
        scales as (z+1)^delta.
        Based on :class:`ZpowerCosmoDistribution`.

        :param Lambda: Local density in units of Gpc^-3
        :type Lambda: float
        :param delta: Index of spatial distribution
        :type delta: float
        :param r_max: Maximum redshift
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        :param luminosity_distribution: Luminosity distribution
        :type luminosity_distribution: :class:`LuminosityDistribution`, optional
        :param is_rate: `True` if modelling a population of transient events,
            `False` if modelling a population of steady-state objects.
            Affects the ``time_adjustment`` method used in cosmo calculations.
            Default is `True`.
        :type is_rate: bool
        """

        spatial_distribution = ZPowerCosmoDistribution(
            seed=seed,
            is_rate=is_rate,
        )

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
        r0: float,
        a: float,
        rise: float,
        decay: float,
        peak: float,
        r_max: float = 5,
        seed: int = 1234,
        luminosity_distribution: LuminosityDistribution = None,
        is_rate: bool = True,
    ):
        """
        A cosmological population with a density that
        scales similarly to the star formation rate.
        Based on :class:`ZpowerCosmoDistribution`.

        :param r0: Local density in units of Gpc^-3
        :type r0: float
        :param a: Offset at z=0
        :type a: float
        :param rise: Rise at low z
        :type rise: float
        :param decay: Decay at high z
        :type decay: float
        :param peak: Peak of z distribution
        :type peak: float
        :param r_max: Maximum redshift
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        :param luminosity_distribution: Luminosity distribution
        :type luminosity_distribution: :class:`LuminosityDistribution`, optional
        :param is_rate: `True` if modelling a population of transient events,
            `False` if modelling a population of steady-state objects.
            Affects the ``time_adjustment`` method used in cosmo calculations.
            Default is `True`.
        :type is_rate: bool
        """

        spatial_distribution = SFRDistribution(seed=seed, is_rate=is_rate)

        spatial_distribution.r0 = r0
        spatial_distribution.a = a
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
        rho: float,
        a: float = 1.64,
        b: float = 4.01,
        R1: float = 0.55,
        R0: float = 8.5,
        r_max: float = 20,
        seed: int = 1234,
        luminosity_distribution: LuminosityDistribution = None,
    ):
        """
        A Milky-way like population based on
        :class:`SpiralGalaxyDistribution`.

        :param rho: Local density
        :type rho: float
        :param a: Shape parameter
        :type a: float
        :param b: Shape parameter
        :type b: float
        :param R1: Scale parameter
        :type R1: float
        :param R0: Scale parameter
        :type R0: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        :param luminosity_distribution: Luminosity distribution
        :type luminosity_distribution: :class:`LuminosityDistribution`, optional
        :param is_rate: `True` if modelling a population of transient events,
            `False` if modelling a population of steady-state objects.
            Affects the ``time_adjustment`` method used in cosmo calculations.
            Default is `True`.
        :type is_rate: bool
        """

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
