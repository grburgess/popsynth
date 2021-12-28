from popsynth.populations.spatial_populations import (
    SphericalPopulation,
    ZPowerSphericalPopulation,
    ZPowerCosmoPopulation,
    SFRPopulation,
)

from popsynth.distributions.schechter_distribution import SchechterDistribution


class SchechterHomogeneousSphericalPopulation(SphericalPopulation):

    def __init__(
        self,
        Lambda: float,
        Lmin: float,
        alpha: float,
        r_max: float = 10,
        seed: int = 1234,
    ):
        """
        A population built on the :class:`ConstantSphericalDistribution`
        spatial distribution and the :class:`SchechterDistribution`
        luminosity distribution.

        :param Lambda: Density per unit volume
        :type Lambda: float
        :param Lmin: Minimum value of the luminosity
        :type Lmin: float
        :param alpha: Index of the luminosity distribution
        :type alpha: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        """

        luminosity_distribution = SchechterDistribution(seed=seed)
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha

        super(SchechterHomogeneousSphericalPopulation, self).__init__(
            Lambda=Lambda,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class SchechterZPowerSphericalPopulation(ZPowerSphericalPopulation):

    def __init__(
        self,
        Lambda: float,
        delta: float,
        Lmin: float,
        alpha: float,
        r_max: float = 10,
        seed: int = 1234,
    ):
        """
        A population built on the :class:`ZPowerSphericalDistribution`
        spatial distribution and the :class:`SchechterDistribution`
        luminosity distribution.

        :param Lambda: Density per unit volume
        :type Lambda: float
        :param delta: Index of the spatial distribution
        :type delta: float
        :param Lmin: Minimum value of the luminosity
        :type Lmin: float
        :param alpha: Index of the luminosity distribution
        :type alpha: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        """

        luminosity_distribution = SchechterDistribution(seed=seed)

        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha

        super(SchechterZPowerSphericalPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class SchechterZPowerCosmoPopulation(ZPowerCosmoPopulation):

    def __init__(
        self,
        Lambda: float,
        delta: float,
        Lmin: float,
        alpha: float,
        r_max: float = 10,
        seed: int = 1234,
        is_rate: bool = True,
    ):
        """
        A population built on the :class:`ZPowerCosmoDistribution`
        spatial distribution and the :class:`SchechterDistribution`
        luminosity distribution.

        :param Lambda: Density in units of Gpc^-3
        :type Lambda: float
        :param delta: Index of the spatial distribution
        :type delta: float
        :param Lmin: Minimum value of the luminosity
        :type Lmin: float
        :param alpha: Index of the luminosity distribution
        :type alpha: float
        :param r_max: Maximum redshift
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        :param is_rate: `True` if modelling a population of transient events,
            `False` if modelling a population of steady-state objects.
            Affects the ``time_adjustment`` method used in cosmo calculations.
            Default is `True`.
        :type is_rate: bool
        """

        luminosity_distribution = SchechterDistribution(seed=seed)

        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha

        super(SchechterZPowerCosmoPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
            is_rate=is_rate,
        )


class SchechterSFRPopulation(SFRPopulation):

    def __init__(
        self,
        r0: float,
        a: float,
        rise: float,
        decay: float,
        peak: float,
        Lmin: float,
        alpha: float,
        r_max: float = 10,
        seed: int = 1234,
        is_rate: bool = True,
    ):
        """
        A population built on the :class:`SFRDistribution`
        spatial distribution and the :class:`SchechterDistribution`
        luminosity distribution.

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
        :param Lmin: Minimum value of the luminosity
        :type Lmin: float
        :param alpha: Index of the luminosity distribution
        :type alpha: float
        :param r_max: Maximum redshift
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        :param is_rate: `True` if modelling a population of transient events,
            `False` if modelling a population of steady-state objects.
            Affects the ``time_adjustment`` method used in cosmo calculations.
            Default is `True`.
        :type is_rate: bool
        """

        luminosity_distribution = SchechterDistribution(seed=seed)

        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha

        super(SchechterSFRPopulation, self).__init__(
            r0=r0,
            a=a,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
            is_rate=is_rate,
        )
