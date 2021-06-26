import numpy as np
from popsynth.distribution import LuminosityDistribution, DistributionParameter


class DeltaDistribution(LuminosityDistribution):

    _distribution_name = "DeltaDistribution"

    Lp = DistributionParameter(vmin=0, default=0)

    def __init__(self, seed: int = 1234, name: str = "delta"):
        """
        A delta function luminosity distribution,
        centred on ``Lp``.

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param Lp: The central value
        :type Lp: :class:`DistributionParameter`
        """
        lf_form = r"\delta(L - L_p)"

        super(DeltaDistribution, self).__init__(
            seed=seed,
            name=name,
            form=lf_form,
        )

    def phi(self, L):

        if isinstance(L, (list, np.ndarray)):

            L = np.array(L)

            out = np.zeros(len(L))

            out[L == self.Lp] = 1

            out[L != self.Lp] = 0

            return out

        else:

            if L == self.Lp:

                return 1

            else:

                return 0

    def draw_luminosity(self, size=1):

        return np.repeat(self.Lp, size)
