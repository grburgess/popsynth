import numpy as np

from popsynth.distribution import LuminosityDistribution, DistributionParameter


class ParetoDistribution(LuminosityDistribution):
    _distribution_name = "ParetoDistribution"

    Lmin = DistributionParameter(default=1, vmin=0)
    alpha = DistributionParameter(default=1)

    def __init__(self, seed: int = 1234, name: str = "pareto"):
        """
        A Pareto luminosity function
        :param seed:
        :param name:
        :returns:
        :rtype:

        """

        lf_form = r"\frac{\alpha L_{\rm min}^{\alpha}}{L^{\alpha+1}}"

        super(ParetoDistribution, self).__init__(seed=seed,
                                                 name=name,
                                                 form=lf_form)

    def phi(self, L):
        """
        The luminosity function

        :param L:
        :returns:
        :rtype:

        """

        out = np.zeros_like(L)

        idx = L >= self.Lmin

        out[idx] = self.alpha * self.Lmin**self.alpha / L[idx]**(self.alpha +
                                                                 1)

        return out

    def draw_luminosity(self, size: int = 1):
        """
        draw the luminosity

        :param size:
        :returns:
        :rtype:

        """

        return (np.random.pareto(self.alpha, size) + 1) * self.Lmin
