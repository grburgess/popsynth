from popsynth.distribution import SpatialDistribution, DistributionParameter


class FlatlandDistribution(SpatialDistribution):
    _distribution_name = "FlatlandDistribution"

    Lambda = DistributionParameter(default=1, vmin=0)

    def __init__(self,
                 seed: int = 1234,
                 name: str = "flatland",
                 form: str = None):
        """
        A flat spatial distribution with only length.

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param form: Mathematical description of distribution
        :type form: str
        :param Lambda: Length
        :type Lambda: :class:`DistributionParameter`
        """
        super(FlatlandDistribution, self).__init__(seed=seed,
                                                   name=name,
                                                   form=form)

    def differential_volume(self, r):

        return 1

    def transform(self, L, r):

        return L

    def dNdV(self, distance):

        return self.Lambda
