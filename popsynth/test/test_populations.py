import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import popsynth
from popsynth import update_logging_level
from popsynth.utils.cosmology import cosmology

update_logging_level("DEBUG")


class DemoSampler(popsynth.AuxiliarySampler):
    _auxiliary_sampler_name = "DemoSampler"
    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=100)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=20, vmin=0)

    def __init__(self):

        super(DemoSampler, self).__init__("demo", observed=False)

    def true_sampler(self, size):

        self._true_values = np.random.normal(self.mu, self.tau, size=size)


class DemoSampler2(popsynth.DerivedLumAuxSampler):
    _auxiliary_sampler_name = "DemoSampler2"
    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=0)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)
    sigma = popsynth.auxiliary_sampler.AuxiliaryParameter(default=0.1, vmin=0)

    def __init__(self):

        super(DemoSampler2, self).__init__("demo2")

    def true_sampler(self, size):

        secondary = self._secondary_samplers["demo"]

        self._true_values = ((np.random.normal(self.mu, self.tau, size=size)) +
                             secondary.true_values -
                             np.log10(1 + self._distance))

    def observation_sampler(self, size):

        self._obs_values = self._true_values + np.random.normal(
            0, self.sigma, size=size)

    def compute_luminosity(self):

        secondary = self._secondary_samplers["demo"]

        return (10**(self._true_values + 54)) / secondary.true_values


class DemoSampler3(popsynth.AuxiliarySampler):
    _auxiliary_sampler_name = "DemoSampler3"
    sigma = popsynth.auxiliary_sampler.AuxiliaryParameter(default=0.1, vmin=0)

    def __init__(self):

        super(DemoSampler3, self).__init__(
            "demo3",
            observed=True,
            uses_distance=True,
            uses_luminosity=True,
        )

    def true_sampler(self, size):

        dl = cosmology.luminosity_distance(self._distance)

        fluxes = self._luminosity / (4 * np.pi * dl**2)

        self._true_values = fluxes

    def observation_sampler(self, size):

        log_fluxes = np.log(self._true_values)

        log_obs_fluxes = log_fluxes + np.random.normal(
            loc=0, scale=self.sigma, size=size)

        self._obs_values = np.exp(log_obs_fluxes)


class DemoSampler4(popsynth.AuxiliarySampler):
    _auxiliary_sampler_name = "DemoSampler4"

    def __init__(self):

        super(DemoSampler4, self).__init__("demo4", observed=False)

    def true_sampler(self, size):

        secondary = self._secondary_samplers["demo3"]

        self._true_values = secondary.obs_values + 10


_spatial_dict = [
    popsynth.populations.SphericalPopulation,
    popsynth.populations.ZPowerSphericalPopulation,
    popsynth.populations.ZPowerCosmoPopulation,
    popsynth.populations.SFRPopulation,
]

_cosmo_dict = [
    popsynth.populations.ZPowerCosmoPopulation,
    popsynth.populations.SFRPopulation,
]

_pareto_dict = [
    popsynth.populations.ParetoHomogeneousSphericalPopulation,
    popsynth.populations.ParetoZPowerSphericalPopulation,
    popsynth.populations.ParetoZPowerCosmoPopulation,
    popsynth.populations.ParetoSFRPopulation,
]

_bpl_dict = [
    popsynth.populations.BPLHomogeneousSphericalPopulation,
    popsynth.populations.BPLZPowerCosmoPopulation,
    popsynth.populations.BPLZPowerSphericalPopulation,
    popsynth.populations.BPLSFRPopulation,
]

_schechter_dict = [
    popsynth.populations.SchechterHomogeneousSphericalPopulation,
    popsynth.populations.SchechterZPowerSphericalPopulation,
    popsynth.populations.SchechterZPowerCosmoPopulation,
    popsynth.populations.SchechterSFRPopulation,
]

_lognorm_dict = [
    popsynth.populations.LogNormalHomogeneousSphericalPopulation,
    popsynth.populations.LogNormalZPowerSphericalPopulation,
    popsynth.populations.LogNormalZPowerCosmoPopulation,
    popsynth.populations.LogNormalSFRPopulation,
]

_log10norm_dict = [
    popsynth.populations.Log10NormalHomogeneousSphericalPopulation,
    popsynth.populations.Log10NormalZPowerSphericalPopulation,
    popsynth.populations.Log10NormalZPowerCosmoPopulation,
    popsynth.populations.Log10NormalSFRPopulation,
]

_spatial_params = [
    dict(Lambda=1.0),
    dict(Lambda=5.0, delta=0.1),
    dict(Lambda=5.0, delta=0.1),
    dict(r0=5.0, a=0.015, rise=0.5, decay=2.0, peak=1.5),
]

_cosmo_params = [
    dict(Lambda=5.0, delta=0.1),
    dict(r0=5.0, a=0.015, rise=0.5, decay=2.0, peak=1.5),
]

_pareto_params = dict(Lmin=2.0, alpha=1.0)

_bpl_params = dict(Lmin=10.0, alpha=-0.5, Lbreak=100, beta=-2.0, Lmax=1000.0)

_lognormal_params = dict(mu=1.0, tau=1.0)


class Popbuilder(object):

    def __init__(self, pop_class, **params):

        self.pop_gen = pop_class(**params)

        self.d1 = DemoSampler()
        self.d2 = DemoSampler2()
        self.d2.set_secondary_sampler(self.d1)
        self.d3 = DemoSampler3()
        self.d4 = DemoSampler4()
        self.d4.set_secondary_sampler(self.d3)

        b = popsynth.BernoulliSelection()
        b.probability = 0.5

        self.d2.set_selection_probability(b)

        for k, v in params.items():

            if k != "is_rate":

                assert k in self.pop_gen._params

    def draw_hard(self):

        s = popsynth.HardFluxSelection()
        s.boundary = 1e-6

        self.pop_gen.set_flux_selection(s)

        pop = self.pop_gen.draw_survey(flux_sigma=0.4, )

        self.reset()

        return pop

    def draw_all(self):

        s = popsynth.UnitySelection()

        self.d2.set_selection_probability(s)

        pop = self.pop_gen.draw_survey()

        assert isinstance(self.pop_gen._flux_selector, popsynth.UnitySelection)

        self.reset()

        return pop

    def draw_none(self):

        s = popsynth.HardFluxSelection()
        s.boundary = 1e999

        self.pop_gen.set_flux_selection(s)

        pop = self.pop_gen.draw_survey(flux_sigma=0.1, )

        self.reset()

        return pop

    def draw_soft(self):

        s = popsynth.SoftFluxSelection()
        s.boundary = 1e-4
        s.strength = 10

        self.pop_gen.set_flux_selection(s)

        pop = self.pop_gen.draw_survey(flux_sigma=0.1, )

        self.reset()

        return pop

    def draw_z_select(self):

        s1 = popsynth.BernoulliSelection()
        s1.probability = 0.5
        s2 = popsynth.SoftFluxSelection()
        s2.boundary = 1e-6
        s2.strength = 10

        self.pop_gen.set_distance_selection(s1)
        self.pop_gen.set_flux_selection(s2)

        pop = self.pop_gen.draw_survey(flux_sigma=0.5)

        self.reset()

        return pop

    def draw_hard_with_selector(self):

        selector = popsynth.HardFluxSelection()
        selector.boundary = 1e-4

        self.pop_gen.set_flux_selection(selector)

        pop = self.pop_gen.draw_survey(flux_sigma=0.5)

        self.reset()

        return pop

    def draw_soft_with_selector(self):

        selector = popsynth.SoftFluxSelection()
        selector.boundary = 1e-4
        selector.strength = 10

        self.pop_gen.set_flux_selection(selector)

        pop = self.pop_gen.draw_survey(flux_sigma=0.5)

        self.reset()

        return pop

    def reset(self):

        add_back = False

        if self.pop_gen._has_derived_luminosity:

            add_back = True

        self.pop_gen.clean(reset=True)

        if add_back:

            print("ADDING BACK")

            # self.d1 = DemoSampler()
            # self.d2 = DemoSampler2()
            # self.d2.set_secondary_sampler(self.d1)

            # b = popsynth.BernoulliSelection()

            #            self.d2.set_selection_probability(b)

            self.pop_gen.add_observed_quantity(self.d2)

        # self.pop_gen._distance_selector_set = False
        # self.pop_gen._flux_selector_set = False
        # self.pop_gen._flux_selector = popsynth.UnitySelection()
        # self.pop_gen._distance_selector = popsynth.UnitySelection()

    def test_it(self):

        self.pop_gen.display()

        self.pop_gen.graph

        #####################

        pop = self.draw_hard()
        pop = self.draw_hard()

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")

        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_soft()
        pop = self.draw_soft()

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")
        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        sub_pop = pop.to_sub_population(observed=True)

        assert sub_pop.n_objects == sub_pop.n_detections

        assert sub_pop.n_objects == sum(pop.selection)

        sub_pop = pop.to_sub_population(observed=False)

        assert sub_pop.n_objects == sub_pop.n_detections

        assert sub_pop.n_objects == sum(~pop.selection)

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_z_select()
        pop = self.draw_z_select()

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")
        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_all()
        pop = self.draw_all()

        pop.to_stan_data()
        pop.display_obs_fluxes()
        pop.display_distances()
        # pop.display_luminosty()
        pop.selected_distances
        pop.selected_fluxes_latent
        pop.selected_fluxes_observed

        self.pop_gen.graph
        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")
        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        pop.graph

        assert sum(~population_reloaded.selection) == 0
        assert population_reloaded.n_objects == population_reloaded.n_detections
        assert population_reloaded.n_non_detections == 0

        os.remove("_saved_pop.h5")

        # clean up

        #####################

        pop = self.draw_none()
        pop = self.draw_none()

        pop.display()

        pop.to_stan_data()
        pop.display_obs_fluxes()
        pop.display_distances()
        #      pop.display_luminosty()
        pop.selected_distances
        pop.selected_fluxes_latent
        pop.selected_fluxes_observed

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")
        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        assert sum(population_reloaded.selection) == 0
        assert (population_reloaded.n_objects ==
                population_reloaded.n_non_detections)
        assert population_reloaded.n_detections == 0

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_hard_with_selector()
        pop = self.draw_hard_with_selector()

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")

        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        os.remove("_saved_pop.h5")

        #####################

        # pop = self.draw_soft_with_selector()
        # pop = self.draw_soft_with_selector()

        # pop.display()

        # fig = pop.display_fluxes()

        # fig = pop.display_flux_sphere()

        # pop.writeto("_saved_pop.h5")
        # population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        # assert population_reloaded.boundary == 1e-4

        # os.remove("_saved_pop.h5")

        # clean up
        plt.close("all")


def test_spatial_population_with_derived():

    for pop, param in zip(_spatial_dict, _spatial_params):

        param = copy.deepcopy(param)

        pb = Popbuilder(pop, **param)

        # first make sure they all fail

        with pytest.raises(RuntimeError):
            pb.draw_hard()
        with pytest.raises(RuntimeError):
            pb.draw_soft()
        with pytest.raises(RuntimeError):
            pb.draw_z_select()

        pb.pop_gen.add_auxiliary_sampler(pb.d2)

        print(len(pb.pop_gen._auxiliary_observations))

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


def test_bpl():

    for pop, param in zip(_bpl_dict, _spatial_params):

        param = copy.deepcopy(param)

        for k, v in _bpl_params.items():

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


def test_non_transient():

    for pop, param in zip(_cosmo_dict, _cosmo_params):

        param = copy.deepcopy(param)

        param["is_rate"] = False

        pb = Popbuilder(pop, **param)

        pb.pop_gen.add_auxiliary_sampler(pb.d2)

        pb.test_it()

        assert not pb.pop_gen.spatial_distribution._is_rate


def test_lumi_and_dist_secondary_sampler():

    for pop, param in zip(_cosmo_dict, _cosmo_params):

        param = copy.deepcopy(param)

        param["is_rate"] = False

        pb = Popbuilder(pop, **param)

        pb.pop_gen.add_auxiliary_sampler(pb.d2)
        pb.pop_gen.add_observed_quantity(pb.d4)

        pb.test_it()
