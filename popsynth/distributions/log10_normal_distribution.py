import numpy as np

from popsynth.distribution import LuminosityDistribution


class Log10NormalDistribution(LuminosityDistribution):
    def __init__(self, mu, tau, seed=1234, name="log10norm"):

        lf_form = r"\frac{\alpha L_{\rm min}^{\alpha}}{L^{\alpha+1}}"

        super(Log10NormalDistribution, self).__init__(
            name=name, seed=seed, form=lf_form
        )

        self._construct_distribution_params(mu=mu, tau=tau)

    def phi(self, L):

        return (1.0 / (self.tau * L * np.sqrt(2 * np.pi))) * np.exp(
            -((np.log10(L) - self.mu) ** 2) / (2 * self.tau ** 2)
        )

    def draw_luminosity(self, size=1):

        x = np.random.normal(loc=self.mu, scale=self.tau, size=size)

        return np.power(10.0, x)

    def __get_mu(self):
        """Calculates the 'mu' property."""
        return self._params["mu"]

    def ___get_mu(self):
        """Indirect accessor for 'mu' property."""
        return self.__get_mu()

    def __set_mu(self, mu):
        """Sets the 'mu' property."""
        self.set_distribution_params(mu=mu, tau=self.tau)

    def ___set_mu(self, mu):
        """Indirect setter for 'mu' property."""
        self.__set_mu(mu)

    mu = property(___get_mu, ___set_mu, doc="""Gets or sets the mu.""")

    def __get_tau(self):
        """Calculates the 'tau' property."""
        return self._params["tau"]

    def ___get_tau(self):
        """Indirect accessor for 'tau' property."""
        return self.__get_tau()

    def __set_tau(self, tau):
        """Sets the 'tau' property."""
        self.set_distribution_params(mu=self.mu, tau=tau)

    def ___set_tau(self, tau):
        """Indirect setter for 'tau' property."""
        self.__set_tau(tau)

    tau = property(___get_tau, ___set_tau, doc="""Gets or sets the tau.""")
