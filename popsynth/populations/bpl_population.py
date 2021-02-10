from popsynth.populations.spatial_populations import (
    SphericalPopulation,
    ZPowerSphericalPopulation,
    ZPowerCosmoPopulation,
    SFRPopulation,
)
from popsynth.distributions.bpl_distribution import BPLDistribution


class BPLHomogeneousSphericalPopulation(SphericalPopulation):
    def __init__(self,
                 Lambda,
                 Lmin,
                 alpha,
                 Lbreak,
                 beta,
                 Lmax,
                 r_max=5,
                 seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param Lmin:
        :param alpha:
        :param Lbreak:
        :param beta:
        :param Lmax:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
    def __init__(self,
                 Lambda,
                 delta,
                 Lmin,
                 alpha,
                 Lbreak,
                 beta,
                 Lmax,
                 r_max=5,
                 seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param delta:
        :param Lmin:
        :param alpha:
        :param Lbreak:
        :param beta:
        :param Lmax:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
        Lambda,
        delta,
        Lmin,
        alpha,
        Lbreak,
        beta,
        Lmax,
        r_max=5,
        seed=1234,
        is_rate=True,
    ):
        """FIXME! briefly describe function

        :param Lambda:
        :param delta:
        :param Lmin:
        :param alpha:
        :param Lbreak:
        :param beta:
        :param Lmax:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
        r0,
        rise,
        decay,
        peak,
        Lmin,
        alpha,
        Lbreak,
        beta,
        Lmax,
        r_max=5,
        seed=1234,
        is_rate=True,
    ):
        """FIXME! briefly describe function

        :param r0:
        :param rise:
        :param decay:
        :param peak:
        :param Lmin:
        :param alpha:
        :param Lbreak:
        :param beta:
        :param Lmax:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

        """

        luminosity_distribution = BPLDistribution(seed=seed)
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha
        luminosity_distribution.Lbreak = Lbreak
        luminosity_distribution.beta = beta
        luminosity_distribution.Lmax = Lmax

        super(BPLSFRPopulation, self).__init__(
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
            is_rate=is_rate,
        )
