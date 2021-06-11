import numpy as np
from popsynth.distribution import LuminosityDistribution, DistributionParameter


class DeltaDistribution(LuminosityDistribution):

    _distribution_name = "DeltaDistribution"

    Lp = DistributionParameter(vmin=0)

    def __init__(self, seed=1234, name="delta"):

        lf_form = r"\delta(L - L_p)"

        super(DeltaDistribution, self).__init__(
            seed=seed,
            name=name,
            form=lf_form,
        )

    def phi(self, L):

        if L == self.Lp:

            return 1

        else:

            return 0

    def draw_luminosity(self, size=1):

        return np.repeat(self.Lp, size)
