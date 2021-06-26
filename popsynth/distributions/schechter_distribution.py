import scipy.special as sf
import numpy as np

from popsynth.distribution import LuminosityDistribution, DistributionParameter


class SchechterDistribution(LuminosityDistribution):
    _distribution_name = "SchechterDistribution"

    Lmin = DistributionParameter(default=1, vmin=0)
    alpha = DistributionParameter(default=1)

    def __init__(self, seed: int = 1234, name: str = "schechter"):
        """
        A Schechter luminosity function as in
        Schechter, Astrophysical Journal, Vol. 203,
        p. 297-306 (1976).

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param Lmin: Minimum value of the luminosity
        :type Lmin: :class:`DistributionParameter`
        :param alpha: Index of the distribution
        :type alpha: :class:`DistributionParameter`
        """
        lf_form = r"\frac{1}{L_{\rm min}^{1+\alpha}\Gamma\left(1+\alpha\right)}"
        lf_form += r" L^{\alpha} \exp\left[ - \frac{L}{L_{\rm min}}\right]"

        super(SchechterDistribution, self).__init__(
            name=name,
            seed=seed,
            form=lf_form,
        )

    def phi(self, L):

        return (L**self.alpha * np.exp(-L / self.Lmin) /
                (self.Lmin**(1 + self.alpha) * sf.gamma(1 + self.alpha)))

    def draw_luminosity(self, size=1):

        xs = 1 - np.random.uniform(size=size)
        return self.Lmin * sf.gammaincinv(1 + self.alpha, xs)
