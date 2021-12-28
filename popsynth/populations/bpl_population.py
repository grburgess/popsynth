from popsynth.populations.spatial_populations import (
    SphericalPopulation,
    ZPowerSphericalPopulation,
    ZPowerCosmoPopulation,
    SFRPopulation,
)
from popsynth.distributions.bpl_distribution import BPLDistribution


class BPLHomogeneousSphericalPopulation(SphericalPopulation):

    def __init__(
        self,
        Lambda: float,
        Lmin: float,
        alpha: float,
        Lbreak: float,
        beta: float,
        Lmax: float,
        r_max: float = 5,
        seed: int = 1234,
    ):
        """
        A population built on the :class:`ConstantSphericalDistribution`
        spatial distribution and the :class:`BPLDistribution`
        luminosity distribution.

        :param Lambda: Density per unit volume
        :type Lambda: float
        :param Lmin: Minimum value of the luminosity
        :type Lmin: float
        :param alpha: Lower luminosity index
        :type alpha: float
        :param Lbreak: Break luminosity
        :type Lbreak: float
        :param beta: Upper luminosity index
        :type beta: float
        :param Lmax: Maximum value of the luminosity
        :type Lmax: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        """
        luminosity_distribution = BPLDistribution(seed=seed)
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha
        luminosity_distribution.Lbreak = Lbreak
        luminosity_distribution.beta = beta
        luminosity_distribution.Lmax = Lmax

        super(BPLHomogeneousSphericalPopulation, self).__init__(
            Lambda=Lambda,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class BPLZPowerSphericalPopulation(ZPowerSphericalPopulation):

    def __init__(
        self,
        Lambda: float,
        delta: float,
        Lmin: float,
        alpha: float,
        Lbreak: float,
        beta: float,
        Lmax: float,
        r_max: float = 5,
        seed: int = 1234,
    ):
        """
        A population built on the :class:`ZPowerSphericalDistribution`
        spatial distribution and the :class:`BPLDistribution`
        luminosity distribution.

        :param Lambda: Density per unit volume
        :type Lambda: float
        :param delta: Index of spatial distribution
        :type delta: float
        :param Lmin: Minimum value of the luminosity
        :type Lmin: float
        :param alpha: Lower luminosity index
        :type alpha: float
        :param Lbreak: Break luminosity
        :type Lbreak: float
        :param beta: Upper luminosity index
        :type beta: float
        :param Lmax: Maximum value of the luminosity
        :type Lmax: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        """
        luminosity_distribution = BPLDistribution(seed=seed)
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha
        luminosity_distribution.Lbreak = Lbreak
        luminosity_distribution.beta = beta
        luminosity_distribution.Lmax = Lmax

        super(BPLZPowerSphericalPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class BPLZPowerCosmoPopulation(ZPowerCosmoPopulation):

    def __init__(
        self,
        Lambda: float,
        delta: float,
        Lmin: float,
        alpha: float,
        Lbreak: float,
        beta: float,
        Lmax: float,
        r_max: float = 5,
        seed: int = 1234,
        is_rate: bool = True,
    ):
        """
        A population built on the :class:`ZPowerCosmoDistribution`
        spatial distribution and the :class:`BPLDistribution`
        luminosity distribution.

        :param Lambda: Density in Gpc^-3
        :type Lambda: float
        :param delta: Index of spatial distribution
        :type delta: float
        :param Lmin: Minimum value of the luminosity
        :type Lmin: float
        :param alpha: Lower luminosity index
        :type alpha: float
        :param Lbreak: Break luminosity
        :type Lbreak: float
        :param beta: Upper luminosity index
        :type beta: float
        :param Lmax: Maximum value of the luminosity
        :type Lmax: float
        :param r_max: Maximum distance
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        """

        luminosity_distribution = BPLDistribution(seed=seed)
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha
        luminosity_distribution.Lbreak = Lbreak
        luminosity_distribution.beta = beta
        luminosity_distribution.Lmax = Lmax

        super(BPLZPowerCosmoPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
            is_rate=is_rate,
        )


class BPLSFRPopulation(SFRPopulation):

    def __init__(
        self,
        r0: float,
        a: float,
        rise: float,
        decay: float,
        peak: float,
        Lmin: float,
        alpha: float,
        Lbreak: float,
        beta: float,
        Lmax: float,
        r_max: float = 5,
        seed: int = 1234,
        is_rate: bool = True,
    ):
        """
        A population built on the :class:`SFRDistribution`
        spatial distribution and the :class:`BPLDistribution`
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
        :param alpha: Lower luminosity index
        :type alpha: float
        :param Lbreak: Break luminosity
        :type Lbreak: float
        :param beta: Upper luminosity index
        :type beta: float
        :param Lmax: Maximum value of the luminosity
        :type Lmax: float
        :param r_max: Maximum redshift
        :type r_max: float
        :param seed: Random seed
        :type seed: int
        """

        luminosity_distribution = BPLDistribution(seed=seed)
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha
        luminosity_distribution.Lbreak = Lbreak
        luminosity_distribution.beta = beta
        luminosity_distribution.Lmax = Lmax

        super(BPLSFRPopulation, self).__init__(
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
