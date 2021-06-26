import numpy as np

from popsynth.distribution import LuminosityDistribution, DistributionParameter


class Log10NormalDistribution(LuminosityDistribution):
    _distribution_name = "Log10NormalDistribution"

    mu = DistributionParameter()
    tau = DistributionParameter(vmin=0)

    def __init__(self, seed: int = 1234, name: str = "log10norm"):
        """
        A log10-normal luminosity function

        Log10Normal(``mu``, ``tau``)

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param mu: Mean of the log10 normal
        :type mu: :class:`DistributionParameter`
        :param tau: Standard deviation of the log10 normal
        :type tau: :class:`DistributionParameter`
        """
        lf_form = r"\mathrm{Log_{10}Normal(\mu, \tau)}"

        super(Log10NormalDistribution, self).__init__(name=name,
                                                      seed=seed,
                                                      form=lf_form)

    def phi(self, L):

        return (1.0 / (self.tau * L * np.sqrt(2 * np.pi))) * np.exp(-(
            (np.log10(L) - self.mu)**2) / (2 * self.tau**2))

    def draw_luminosity(self, size=1):

        x = np.random.normal(loc=self.mu, scale=self.tau, size=size)

        return np.power(10.0, x)
