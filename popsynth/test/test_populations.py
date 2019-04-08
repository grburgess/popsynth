import popsynth
import pytest


class DemoSampler(popsynth.AuxiliarySampler):
    def __init__(self, mu=2, tau=1.0, sigma=1):

        self._mu = mu
        self._tau = tau

        super(DemoSampler, self).__init__("demo", sigma, observed=False)

    def true_sampler(self, size):

        self._true_values = np.random.normal(self._mu, self._tau, size=size)


class DemoSampler2(popsynth.AuxiliarySampler):
    def __init__(self, mu=2, tau=1.0, sigma=1):

        self._mu = mu
        self._tau = tau

        super(DemoSampler2, self).__init__("demo2", sigma, observed=True)

    def true_sampler(self, size):

        secondary = self._secondary_samplers["demo"]

        self._true_values = (
            (np.random.normal(self._mu, self._tau, size=size))
            + secondary.true_values
            - np.log10(1 + self._distance)
        )

    def observation_sampler(self, size):

        self._obs_values = self._true_values + np.random.normal(
            0, self._sigma, size=size
        )


_spatial_dict = [
    popsynth.populations.SphericalPopulation,
    popsynth.populations.ZPowerSphericalPopulation,
    popsynth.populations.SFRPopulation,
]

_spatial_params = [
    dict(Lambda=10.0),
    dict(Lambda=10.0, delta=0.1),
    dict(r0=100, rise=0.5, decay=2.0, peak=1.5),
]
_pareto_params = dict(Lmin=2.0, alpha=1.0)
_lognormal_params = dict(mu=1.0, tau=1.0)


class Popbuilder(object):
    def __init__(self, pop_class, **params):

        self.pop_gen = pop_class(**params)

        for k, v in params.items():

            assert k in self.pop_gen._params

    def draw_hard(self):

        return self.pop_gen.draw_survey(boundary=1e-10, flux_sigma=0.1, hard_cut=True)

    def draw_soft(self):

        return self.pop_gen.draw_survey(boundary=1e-10, flux_sigma=0.1, hard_cut=False)

    def draw_z_select(self):

        return self.pop_gen.draw_survey(
            boundary=1e-10, flux_sigma=0.1, distance_probability=0.5
        )


def test_spatial_population_with_derived():

    # first make sure they all fail
    for pop, param in zip(_spatial_dict, _spatial_params):

        pb = Popbuilder(pop, **param)

        with pytest.raises(AssertionError):
            pb.draw_hard()
        with pytest.raises(AssertionError):
            pb.draw_soft()
        with pytest.raises(AssertionError):
            pb.draw_z_select()
