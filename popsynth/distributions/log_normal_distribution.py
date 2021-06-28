import numpy as np

from popsynth.distribution import LuminosityDistribution, DistributionParameter


class LogNormalDistribution(LuminosityDistribution):
    _distribution_name = "LogNormalDistribution"

    mu = DistributionParameter()
    tau = DistributionParameter(vmin=0)

    def __init__(self, seed: int = 1234, name: str = "lognorm"):
        """
        A log-normal luminosity distribution.

        LogNormal(``mu``, ``tau``)

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param mu: Mean of the log normal
        :type mu: :class:`DistributionParameter`
        :param tau: Standard deviation of the log normal
        :type tau: :class:`DistributionParameter`
        """
        lf_form = r"\mathrm{LogNorm}(\mu, \tau)"

        super(LogNormalDistribution, self).__init__(
            name=name,
            seed=seed,
            form=lf_form,
        )

    def phi(self, L):

        return (1.0 / (self.tau * L * np.sqrt(2 * np.pi))) * np.exp(-(
            (np.log(L) - self.mu)**2) / (2 * self.tau**2))

    def draw_luminosity(self, size=1):

        return np.random.lognormal(mean=self.mu, sigma=self.tau, size=size)
