import numpy as np
import scipy.stats as stats

from popsynth.distribution import LuminosityDistribution, DistributionParameter


class LogNormalDistribution(LuminosityDistribution):

    mu = DistributionParameter()
    tau = DistributionParameter(vmin=0)

    def __init__(self, seed=1234, name="lognorm"):

        lf_form = r"\frac{\alpha L_{\rm min}^{\alpha}}{L^{\alpha+1}}"

        super(LogNormalDistribution, self).__init__(name=name,
                                                    seed=seed,
                                                    form=lf_form)

    def phi(self, L):

        return (1.0 / (self.tau * L * np.sqrt(2 * np.pi))) * np.exp(-(
            (np.log(L) - self.mu)**2) / (2 * self.tau**2))

    def draw_luminosity(self, size=1):

        return np.random.lognormal(mean=self.mu, sigma=self.tau, size=size)
