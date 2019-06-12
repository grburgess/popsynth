import popsynth
import pytest
import numpy as np
import os
import copy
import matplotlib.pyplot as plt


class DemoSampler(popsynth.AuxiliarySampler):
    def __init__(self, mu=2, tau=1.0, sigma=1):

        self._mu = mu
        self._tau = tau

        truth = dict(mu=mu, tau=tau)

        super(DemoSampler, self).__init__("demo", sigma, observed=False, truth=truth)

    def true_sampler(self, size):

        self._true_values = np.random.normal(self._mu, self._tau, size=size)


class DemoSampler2(popsynth.DerivedLumAuxSampler):
    def __init__(self, mu=2, tau=1.0, sigma=1):

        self._mu = mu
        self._tau = tau

        truth = dict(mu=mu, tau=tau)

        super(DemoSampler2, self).__init__("demo2", sigma, observed=True, truth=truth)

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

    def compute_luminosity(self):

        secondary = self._secondary_samplers["demo"]

        return (10 ** (self._true_values + 54)) / secondary.true_values


_spatial_dict = [
    popsynth.populations.SphericalPopulation,
    popsynth.populations.ZPowerSphericalPopulation,
    popsynth.populations.SFRPopulation,
    popsynth.populations.ZPowerCosmoPopulation,
]

_pareto_dict = [
    popsynth.populations.ParetoHomogeneousSphericalPopulation,
    popsynth.populations.ParetoZPowerSphericalPopulation,
    popsynth.populations.ParetoSFRPopulation,
]


_schechter_dict = [
    popsynth.populations.SchechterHomogeneousSphericalPopulation,
    popsynth.populations.SchechterZPowerSphericalPopulation,
    popsynth.populations.SchechterSFRPopulation,
]


_lognorm_dict = [
    popsynth.populations.LogNormalHomogeneousSphericalPopulation,
    popsynth.populations.LogNormalZPowerSphericalPopulation,
    popsynth.populations.LogNormalSFRPopulation,
]


_log10norm_dict = [
    popsynth.populations.Log10NormalHomogeneousSphericalPopulation,
    popsynth.populations.Log10NormalZPowerSphericalPopulation,
    popsynth.populations.Log10NormalSFRPopulation,
]


_spatial_params = [
    dict(Lambda=1.0),
    dict(Lambda=5.0, delta=0.1),
    dict(r0=5.0, rise=0.5, decay=2.0, peak=1.5),
]
_pareto_params = dict(Lmin=2.0, alpha=1.0)

_lognormal_params = dict(mu=1.0, tau=1.0)


class Popbuilder(object):
    def __init__(self, pop_class, **params):

        self.pop_gen = pop_class(**params)

        self.d1 = DemoSampler(100, 20, 0.1)
        self.d2 = DemoSampler2(0, 1, 0.1)
        self.d2.set_secondary_sampler(self.d1)

        for k, v in params.items():

            assert k in self.pop_gen._params

    def draw_hard(self):

        return self.pop_gen.draw_survey(
            boundary=1e-6, flux_sigma=0.4, hard_cut=True, verbose=False
        )

    def draw_soft(self):

        return self.pop_gen.draw_survey(boundary=1e-4, flux_sigma=0.1, hard_cut=False)

    def draw_z_select(self):

        return self.pop_gen.draw_survey(
            boundary=1e-6, flux_sigma=0.5, distance_probability=0.5
        )

    def test_it(self):

        self.pop_gen.display()

        #####################

        pop = self.draw_hard()

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")

        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_soft()

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")
        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_z_select()

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")
        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        os.remove("_saved_pop.h5")

        # clean up
        plt.close("all")


def test_spatial_population_with_derived():

    for pop, param in zip(_spatial_dict, _spatial_params):

        param = copy.deepcopy(param)

        pb = Popbuilder(pop, **param)

        # first make sure they all fail

        with pytest.raises(AssertionError):
            pb.draw_hard()
        with pytest.raises(AssertionError):
            pb.draw_soft()
        with pytest.raises(AssertionError):
            pb.draw_z_select()

        pb.pop_gen.add_observed_quantity(pb.d2)

        pb.test_it()


def test_pareto():

    for pop, param in zip(_pareto_dict, _spatial_params):

        param = copy.deepcopy(param)

        for k, v in _pareto_params.items():

            param[k] = v

        pb = Popbuilder(pop, **param)

        pb.test_it()

        pb.pop_gen.add_observed_quantity(pb.d2)

        pb.test_it()


def test_schecter():

    for pop, param in zip(_schechter_dict, _spatial_params):

        param = copy.deepcopy(param)

        for k, v in _pareto_params.items():

            param[k] = v

        pb = Popbuilder(pop, **param)

        pb.test_it()

        pb.pop_gen.add_observed_quantity(pb.d2)

        pb.test_it()


def test_lnorm():

    for pop, param in zip(_lognorm_dict, _spatial_params):

        param = copy.deepcopy(param)

        for k, v in _lognormal_params.items():

            param[k] = v

        pb = Popbuilder(pop, **param)

        pb.test_it()

        pb.pop_gen.add_observed_quantity(pb.d2)

        pb.test_it()


def test_l10norm():

    for pop, param in zip(_log10norm_dict, _spatial_params):

        param = copy.deepcopy(param)

        for k, v in _lognormal_params.items():

            param[k] = v

        pb = Popbuilder(pop, **param)

        pb.test_it()

        pb.pop_gen.add_observed_quantity(pb.d2)

        pb.test_it()
