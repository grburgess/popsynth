import numpy as np

from popsynth.distribution import SpatialDistribution


class SphericalDistribution(SpatialDistribution):
    def __init__(self, r_max=10, seed=1234, name="sphere", form=None, truth={}):

        super(SphericalDistribution, self).__init__(
            r_max=r_max, seed=seed, name=name, form=form, truth=truth
        )

    def differential_volume(self, r):

        return 4 * np.pi * r * r

    def transform(self, L, r):

        return L / (4.0 * np.pi * r * r)


class ConstantSphericalDistribution(SphericalDistribution):
    def __init__(
        self,
        Lambda=1.0,
        r_max=10.0,
        seed=1234,
        name="cons_sphere",
        form=None,
        truth=None,
    ):

        if form is None:
            form = r"\Lambda"

        if truth is None:

            truth = dict(Lambda=Lambda)

        super(ConstantSphericalDistribution, self).__init__(
            r_max=r_max, seed=seed, name=name, form=form, truth=truth
        )

        self._construct_distribution_params(Lambda=Lambda)

    def __get_Lambda(self):
        """Calculates the 'Lambda' property."""
        return self._params["Lambda"]

    def ___get_Lambda(self):
        """Indirect accessor for 'Lambda' property."""
        return self.__get_Lambda()

    def __set_Lambda(self, Lambda):
        """Sets the 'Lambda' property."""
        self.set_distribution_params(Lambda=Lambda)

    def ___set_Lambda(self, Lambda):
        """Indirect setter for 'Lambda' property."""
        self.__set_Lambda(Lambda)

    Lambda = property(___get_Lambda, ___set_Lambda, doc="""Gets or sets the Lambda.""")

    def dNdV(self, distance):

        return self._params["Lambda"]


class ZPowerSphericalDistribution(ConstantSphericalDistribution):
    def __init__(
        self, Lambda=1.0, delta=1.0, r_max=10.0, seed=1234, name="zpow_sphere"
    ):

        spatial_form = r"\Lambda (z+1)^{\delta}"

        truth = dict(Lambda=Lambda, delta=delta)

        super(ZPowerSphericalDistribution, self).__init__(
            Lambda, r_max, seed, name, form=spatial_form, truth=truth
        )

        self._construct_distribution_params(Lambda=Lambda, delta=delta)

    def __get_delta(self):
        """Calculates the 'delta' property."""
        return self._params["delta"]

    def ___get_delta(self):
        """Indirect accessor for 'delta' property."""
        return self.__get_delta()

    def __set_delta(self, delta):
        """Sets the 'delta' property."""
        self.set_distribution_params(delta=delta)

    def ___set_delta(self, delta):
        """Indirect setter for 'delta' property."""
        self.__set_delta(delta)

    delta = property(___get_delta, ___set_delta, doc="""Gets or sets the delta.""")

    def dNdV(self, distance):

        return self._params["Lambda"] * np.power(distance + 1.0, self._params["delta"])
