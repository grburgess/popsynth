import numpy as np

from popsynth.distribution import LuminosityDistribution, DistributionParameter


class Log10NormalDistribution(LuminosityDistribution):

    mu = DistributionParameter()
    tau = DistributionParameter(vmin=0)

    def __init__(self, seed=1234, name="log10norm"):

        lf_form = r"\frac{\alpha L_{\rm min}^{\alpha}}{L^{\alpha+1}}"

        super(Log10NormalDistribution, self).__init__(name=name,
                                                      seed=seed,
                                                      form=lf_form)

    def phi(self, L):

        return (1.0 / (self.tau * L * np.sqrt(2 * np.pi))) * np.exp(-(
            (np.log10(L) - self.mu)**2) / (2 * self.tau**2))

    def draw_luminosity(self, size=1):

        x = np.random.normal(loc=self.mu, scale=self.tau, size=size)

        return np.power(10.0, x)
