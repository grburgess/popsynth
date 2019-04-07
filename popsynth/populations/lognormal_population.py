from popsynth.populations.spatial_populations import SphericalPopulation, ZPowerSphericalPopulation, SFRPopulation

from popsynth.distributions.log10_normal_distribution import Log10NormalDistribution
from popsynth.distributions.log_normal_distribution import LogNormalDistribution


class LogNormalHomogeneousSphericalPopulation(SphericalPopulation):
    def __init__(self, Lambda, mu, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda: 
        :param mu: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        luminosity_distribution = LogNormalDistribution(mu=mu, alpha=alpha, seed=seed)

        super(LogNormalHomogeneousSphericalPopulation, self).__init__(
            Lambda=Lambda,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution)

class LogNormalZPowerSphericalPopulation(ZPowerSphericalPopulation):
    def __init__(self, Lambda, delta, mu, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda: 
        :param delta:
        :param mu: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        luminosity_distribution = LogNormalDistribution(mu=mu, alpha=alpha, seed=seed)

        super(LogNormalHomogeneousSphericalPopulation, self).__init__(
            Lambda=Lambda,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution)


        
class LogNormalSFRPopulation(SphericalPopulation):
    def __init__(self, r0, rise, decay, peak, mu, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param r0: 
        :param rise: 
        :param decay: 
        :param peak: 
         :param mu: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """


        luminosity_distribution = LogNormalDistribution(mu=mu, alpha=alpha, seed=seed)

        super(LogNormalSFRPopulation, self).__init__(
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution)

class Log10NormalHomogeneousSphericalPopulation(SphericalPopulation):
    def __init__(self, Lambda, mu, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda: 
        :param mu: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        luminosity_distribution = Log10NormalDistribution(mu=mu, alpha=alpha, seed=seed)

        super(Log10NormalHomogeneousSphericalPopulation, self).__init__(
            Lambda=Lambda,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution)

class Log10NormalZPowerSphericalPopulation(ZPowerSphericalPopulation):
    def __init__(self, Lambda, delta, mu, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda: 
        :param delta:
        :param mu: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        luminosity_distribution = Log10NormalDistribution(mu=mu, alpha=alpha, seed=seed)

        super(Log10NormalHomogeneousSphericalPopulation, self).__init__(
            Lambda=Lambda,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution)


        
class Log10NormalSFRPopulation(SphericalPopulation):
    def __init__(self, r0, rise, decay, peak, mu, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param r0: 
        :param rise: 
        :param decay: 
        :param peak: 
        :param mu: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """


        luminosity_distribution = Log10NormalDistribution(mu=mu, alpha=alpha, seed=seed)

        super(Log10NormalSFRPopulation, self).__init__(
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            luminosity_distribution=luminosity_distribution)

