from popsynth.populations.spatial_populations import (
    SphericalPopulation,
    ZPowerSphericalPopulation,
    ZPowerCosmoPopulation,
    SFRPopulation,
)

from popsynth.distributions.schechter_distribution import SchechterDistribution


class SchechterHomogeneousSphericalPopulation(SphericalPopulation):
    def __init__(self, Lambda, Lmin, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param Lmin:
        :param alpha:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
    def __init__(self, Lambda, delta, Lmin, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param delta:
        :param Lmin:
        :param alpha:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
        Lambda,
        delta,
        Lmin,
        alpha,
        r_max=10,
        seed=1234,
        is_rate=True,
    ):
        """FIXME! briefly describe function

        :param Lambda:
        :param delta:
        :param Lmin:
        :param alpha:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
        r0,
        rise,
        decay,
        peak,
        Lmin,
        alpha,
        r_max=10,
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
        :param r_max:
        :param seed:
        :returns:
        :rtype:

        """

        luminosity_distribution = SchechterDistribution(seed=seed)

        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha

        super(SchechterSFRPopulation, self).__init__(
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
            is_rate=is_rate,
        )
