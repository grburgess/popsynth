from popsynth.populations.spatial_populations import (
    SphericalPopulation,
    ZPowerSphericalPopulation,
    ZPowerCosmoPopulation,
    SFRPopulation,
)
from popsynth.distributions.pareto_distribution import ParetoDistribution


class ParetoHomogeneousSphericalPopulation(SphericalPopulation):
    def __init__(self, Lambda, Lmin, alpha, r_max=5, seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param Lmin:
        :param alpha:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

        """

        luminosity_distribution = ParetoDistribution(seed=seed)
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha

        super(ParetoHomogeneousSphericalPopulation, self).__init__(
            Lambda=Lambda,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class ParetoZPowerSphericalPopulation(ZPowerSphericalPopulation):
    def __init__(self, Lambda, delta, Lmin, alpha, r_max=5, seed=1234):
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

        luminosity_distribution = ParetoDistribution(seed=seed)
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha

        super(ParetoZPowerSphericalPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class ParetoZPowerCosmoPopulation(ZPowerCosmoPopulation):
    def __init__(self, Lambda, delta, Lmin, alpha, r_max=5, seed=1234):
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

        luminosity_distribution = ParetoDistribution(seed=seed)
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha

        super(ParetoZPowerCosmoPopulation, self).__init__(
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class ParetoSFRPopulation(SFRPopulation):
    def __init__(self, r0, rise, decay, peak, Lmin, alpha, r_max=5, seed=1234):
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

        luminosity_distribution = ParetoDistribution(seed=seed)
        luminosity_distribution.Lmin = Lmin
        luminosity_distribution.alpha = alpha

        super(ParetoSFRPopulation, self).__init__(
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )
