from popsynth.populations.spatial_populations import (
    SphericalPopulation,
    ZPowerSphericalPopulation,
    ZPowerCosmoPopulation,
    SFRPopulation,
)

from popsynth.distributions.log10_normal_distribution import Log10NormalDistribution
from popsynth.distributions.log_normal_distribution import LogNormalDistribution


class LogNormalHomogeneousSphericalPopulation(SphericalPopulation):
    def __init__(self, Lambda, mu, tau, r_max=5, seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param mu:
        :param tau:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
    def __init__(self, Lambda, delta, mu, tau, r_max=5, seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param delta:
        :param mu:
        :param tau:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
    def __init__(self, Lambda, delta, mu, tau, r_max=5, seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param delta:
        :param mu:
        :param tau:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
        )


class LogNormalSFRPopulation(SFRPopulation):
    def __init__(self, r0, rise, decay, peak, mu, tau, r_max=5, seed=1234):
        """FIXME! briefly describe function

        :param r0:
        :param rise:
        :param decay:
        :param peak:
         :param mu:
        :param tau:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

        """

        luminosity_distribution = LogNormalDistribution(seed=seed)
        luminosity_distribution.mu = mu
        luminosity_distribution.tau = tau

        super(LogNormalSFRPopulation, self).__init__(
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )


class Log10NormalHomogeneousSphericalPopulation(SphericalPopulation):
    def __init__(self, Lambda, mu, tau, r_max=5, seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param mu:
        :param tau:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
    def __init__(self, Lambda, delta, mu, tau, r_max=5, seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param delta:
        :param mu:
        :param tau:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
    def __init__(self, Lambda, delta, mu, tau, r_max=5, seed=1234):
        """FIXME! briefly describe function

        :param Lambda:
        :param delta:
        :param mu:
        :param tau:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

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
        )


class Log10NormalSFRPopulation(SFRPopulation):
    def __init__(self, r0, rise, decay, peak, mu, tau, r_max=5, seed=1234):
        """FIXME! briefly describe function

        :param r0:
        :param rise:
        :param decay:
        :param peak:
        :param mu:
        :param tau:
        :param r_max:
        :param seed:
        :returns:
        :rtype:

        """

        luminosity_distribution = Log10NormalDistribution(seed=seed)
        luminosity_distribution.mu = mu
        luminosity_distribution.tau = tau

        super(Log10NormalSFRPopulation, self).__init__(
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution,
        )
