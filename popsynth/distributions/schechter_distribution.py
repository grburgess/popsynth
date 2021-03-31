import scipy.special as sf
import numpy as np

from popsynth.distribution import LuminosityDistribution, DistributionParameter


class SchechterDistribution(LuminosityDistribution):

    Lmin = DistributionParameter(default=1, vmin=0)
    alpha = DistributionParameter(default=1)

    def __init__(self, seed=1234, name="schechter"):
        """
        A Schechter luminosity function
        :param seed: the random number seed
        :param name: the name
        :returns: None
        :rtype:

        """

        form = r"\frac{1}{L_{\rm min}^{1+\alpha} \Gamma\left(1+\alpha\right)} L^{\alpha} \exp\left[ - \frac{L}{L_{\rm min}}\right]"

        super(SchechterDistribution, self).__init__(name=name,
                                                    seed=seed,
                                                    form=form)

    def phi(self, L):
        """FIXME! briefly describe function

        :param L:
        :returns:
        :rtype:

        """

        return (L**self.alpha * np.exp(-L / self.Lmin) /
                (self.Lmin**(1 + self.alpha) * sf.gamma(1 + self.alpha)))

    def draw_luminosity(self, size=1):
        """FIXME! briefly describe function

        :param size:
        :returns:
        :rtype:

        """

        xs = 1 - np.random.uniform(size=size)
        return self.Lmin * sf.gammaincinv(1 + self.alpha, xs)
