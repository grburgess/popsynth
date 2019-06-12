import numpy as np

from popsynth.distribution import LuminosityDistribution


class ParetoDistribution(LuminosityDistribution):
    def __init__(self, Lmin, alpha, seed=1234, name="pareto"):
        """
        A Pareto luminosity function

        :param Lmin:
        :param alpha:
        :param r_max:
        :param seed:
        :param name:
        :returns:
        :rtype:

        """

        truth = dict(Lmin=Lmin, alpha=alpha)

        lf_form = r"\frac{\alpha L_{\rm min}^{\alpha}}{L^{\alpha+1}}"
        super(ParetoDistribution, self).__init__(
            seed=seed, name=name, form=lf_form, truth=truth
        )

        self._construct_distribution_params(Lmin=Lmin, alpha=alpha)

    def phi(self, L):
        """
        The luminosity function

        :param L:
        :returns:
        :rtype:

        """

        out = np.zeros_like(L)

        idx = L >= self.Lmin

        out[idx] = self.alpha * self.Lmin ** self.alpha / L[idx] ** (self.alpha + 1)

        return out

    def draw_luminosity(self, size=1):
        """FIXME! briefly describe function

        :param size:
        :returns:
        :rtype:

        """

        return (np.random.pareto(self._params["alpha"], size) + 1) * self._params[
            "Lmin"
        ]

    def __get_Lmin(self):
        """Calculates the 'Lmin' property."""
        return self._params["Lmin"]

    def ___get_Lmin(self):
        """Indirect accessor for 'Lmin' property."""
        return self.__get_Lmin()

    def __set_Lmin(self, Lmin):
        """Sets the 'Lmin' property."""
        self.set_distribution_params(alpha=self.alpha, Lmin=Lmin)

    def ___set_Lmin(self, Lmin):
        """Indirect setter for 'Lmin' property."""
        self.__set_Lmin(Lmin)

    Lmin = property(___get_Lmin, ___set_Lmin, doc="""Gets or sets the Lmin.""")

    def __get_alpha(self):
        """Calculates the 'alpha' property."""
        return self._params["alpha"]

    def ___get_alpha(self):
        """Indirect accessor for 'alpha' property."""
        return self.__get_alpha()

    def __set_alpha(self, alpha):
        """Sets the 'alpha' property."""
        self.set_distribution_params(alpha=alpha, Lmin=self.Lmin)

    def ___set_alpha(self, alpha):
        """Indirect setter for 'alpha' property."""
        self.__set_alpha(alpha)

    alpha = property(___get_alpha, ___set_alpha, doc="""Gets or sets the alpha.""")
