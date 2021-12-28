from popsynth.populations.spatial_populations import (
    SphericalPopulation,
    ZPowerSphericalPopulation,
    ZPowerCosmoPopulation,
    SFRPopulation,
)

from popsynth.distributions.log10_normal_distribution import Log10NormalDistribution
from popsynth.distributions.log_normal_distribution import LogNormalDistribution


class LogNormalHomogeneousSphericalPopulation(SphericalPopulation):

    def __init__(
        self,
        Lambda: float,
        mu: float,
        tau: float,
        r_max: float = 5,
        seed: int = 1234,
    ):
        """
        A population built on the :class:`ConstantSphericalDistribution`
        spatial distribution and the :class:`LogNormalDistribution`
        luminosity distribution.

        :param Lambda: Density per unit volume
        :type Lambda: float
        :param mu: Mean of the luminosity distribution
        :type mu: float
        :param tau: Standard deviation of the luminosity distribution
        :type tau: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        """

        luminosity_distribution = LogNormalDistribution(seed=seed)
        luminosity_distribution.mu = mu
        luminosity_distribution.tau = tau

        super(LogNormalHomogeneousSphericalPopulation, self).__init__(
            Lambda=Lambda,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class LogNormalZPowerSphericalPopulation(ZPowerSphericalPopulation):

    def __init__(
        self,
        Lambda: float,
        delta: float,
        mu: float,
        tau: float,
        r_max: float = 5,
        seed: int = 1234,
    ):
        """
        A population built on the :class:`ZPowerSphericalDistribution`
        spatial distribution and the :class:`LogNormalDistribution`
        luminosity distribution.

        :param Lambda: Density per unit volume
        :type Lambda: float
        :param delta: Index of the spatial distribution
        :type delta: float
        :param mu: Mean of the luminosity distribution
        :type mu: float
        :param tau: Standard deviation of the luminosity distribution
        :type tau: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        """

        luminosity_distribution = LogNormalDistribution(seed=seed)
        luminosity_distribution.mu = mu
        luminosity_distribution.tau = tau

        super(LogNormalZPowerSphericalPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class LogNormalZPowerCosmoPopulation(ZPowerCosmoPopulation):

    def __init__(
        self,
        Lambda: float,
        delta: float,
        mu: float,
        tau: float,
        r_max: float = 5,
        seed: int = 1234,
        is_rate: bool = True,
    ):
        """
        A population built on the :class:`ZPowerCosmoDistribution`
        spatial distribution and the :class:`LogNormalDistribution`
        luminosity distribution.

        :param Lambda: Density in units of Gpc^-3
        :type Lambda: float
        :param delta: Index of the spatial distribution
        :type delta: float
        :param mu: Mean of the luminosity distribution
        :type mu: float
        :param tau: Standard deviation of the luminosity distribution
        :type tau: float
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

        luminosity_distribution = LogNormalDistribution(seed=seed)
        luminosity_distribution.mu = mu
        luminosity_distribution.tau = tau

        super(LogNormalZPowerCosmoPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
            is_rate=is_rate,
        )


class LogNormalSFRPopulation(SFRPopulation):

    def __init__(
        self,
        r0: float,
        a: float,
        rise: float,
        decay: float,
        peak: float,
        mu: float,
        tau: float,
        r_max: float = 5,
        seed: int = 1234,
        is_rate: bool = True,
    ):
        """
        A population built on the :class:`SFRDistribution`
        spatial distribution and the :class:`LogNormalDistribution`
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
        :param mu: Mean of the luminosity distribution
        :type mu: float
        :param tau: Standard deviation of the luminosity distribution
        :type tau: float
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

        luminosity_distribution = LogNormalDistribution(seed=seed)
        luminosity_distribution.mu = mu
        luminosity_distribution.tau = tau

        super(LogNormalSFRPopulation, self).__init__(
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


class Log10NormalHomogeneousSphericalPopulation(SphericalPopulation):

    def __init__(
        self,
        Lambda: float,
        mu: float,
        tau: float,
        r_max: float = 5,
        seed: int = 1234,
    ):
        """
        A population built on the :class:`ConstantSphericalDistribution`
        spatial distribution and the :class:`Log10NormalDistribution`
        luminosity distribution.

        :param Lambda: Density per unit volume
        :type Lambda: float
        :param mu: Mean of the luminosity distribution
        :type mu: float
        :param tau: Standard deviation of the luminosity distribution
        :type tau: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        """

        luminosity_distribution = Log10NormalDistribution(seed=seed)
        luminosity_distribution.mu = mu
        luminosity_distribution.tau = tau

        super(Log10NormalHomogeneousSphericalPopulation, self).__init__(
            Lambda=Lambda,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class Log10NormalZPowerSphericalPopulation(ZPowerSphericalPopulation):

    def __init__(
        self,
        Lambda: float,
        delta: float,
        mu: float,
        tau: float,
        r_max: float = 5,
        seed: int = 1234,
    ):
        """
        A population built on the :class:`ZPowerSphericalDistribution`
        spatial distribution and the :class:`Log10NormalDistribution`
        luminosity distribution.

        :param Lambda: Density per unit volume
        :type Lambda: float
        :param delta: Index of the spatial distribution
        :type delta: float
        :param mu: Mean of the luminosity distribution
        :type mu: float
        :param tau: Standard deviation of the luminosity distribution
        :type tau: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        """

        luminosity_distribution = Log10NormalDistribution(seed=seed)
        luminosity_distribution.mu = mu
        luminosity_distribution.tau = tau

        super(Log10NormalZPowerSphericalPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class Log10NormalZPowerCosmoPopulation(ZPowerCosmoPopulation):

    def __init__(
        self,
        Lambda: float,
        delta: float,
        mu: float,
        tau: float,
        r_max: float = 5,
        seed: float = 1234,
        is_rate: float = True,
    ):
        """
        A population built on the :class:`ZPowerCosmoDistribution`
        spatial distribution and the :class:`Log10NormalDistribution`
        luminosity distribution.

        :param Lambda: Density in units of Gpc^-3
        :type Lambda: float
        :param delta: Index of the spatial distribution
        :type delta: float
        :param mu: Mean of the luminosity distribution
        :type mu: float
        :param tau: Standard deviation of the luminosity distribution
        :type tau: float
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

        luminosity_distribution = Log10NormalDistribution(seed=seed)
        luminosity_distribution.mu = mu
        luminosity_distribution.tau = tau

        super(Log10NormalZPowerCosmoPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
            is_rate=is_rate,
        )


class Log10NormalSFRPopulation(SFRPopulation):

    def __init__(
        self,
        r0: float,
        a: float,
        rise: float,
        decay: float,
        peak: float,
        mu: float,
        tau: float,
        r_max: float = 5,
        seed: int = 1234,
        is_rate: bool = True,
    ):
        """
        A population built on the :class:`SFRDistribution`
        spatial distribution and the :class:`Log10NormalDistribution`
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
        :param mu: Mean of the luminosity distribution
        :type mu: float
        :param tau: Standard deviation of the luminosity distribution
        :type tau: float
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

        luminosity_distribution = Log10NormalDistribution(seed=seed)
        luminosity_distribution.mu = mu
        luminosity_distribution.tau = tau

        super(Log10NormalSFRPopulation, self).__init__(
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
