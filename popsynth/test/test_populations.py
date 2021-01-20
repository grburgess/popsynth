import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

import popsynth


class DemoSampler(popsynth.AuxiliarySampler):

    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=100)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=20, vmin=0)

    def __init__(self):

        super(DemoSampler, self).__init__("demo", observed=False)

    def true_sampler(self, size):

        self._true_values = np.random.normal(self.mu, self.tau, size=size)


class DemoSampler2(popsynth.DerivedLumAuxSampler):

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


_spatial_dict = [
    popsynth.populations.SphericalPopulation,
    popsynth.populations.ZPowerSphericalPopulation,
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
    dict(r0=5.0, rise=0.5, decay=2.0, peak=1.5),
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

        b = popsynth.BernoulliSelection(probability=0.5)

        self.d2.set_selection_probability(b)

        for k, v in params.items():

            assert k in self.pop_gen._params

    def draw_hard(self, verbose):

        pop = self.pop_gen.draw_survey(boundary=1e-6,
                                       flux_sigma=0.4,
                                       hard_cut=True,
                                       verbose=verbose)

        self.reset()

        return pop

    def draw_all(self, verbose):

        pop = self.pop_gen.draw_survey(
            boundary=1e-999,
            flux_sigma=0.1,
            hard_cut=True,
            verbose=verbose,
            no_selection=True,
        )

        self.reset()

        return pop

    def draw_none(self, verbose):

        pop = self.pop_gen.draw_survey(
            boundary=1e999,
            flux_sigma=0.1,
            hard_cut=True,
            verbose=verbose,
            no_selection=False,
        )

        self.reset()

        return pop

    def draw_soft(self, verbose):

        pop = self.pop_gen.draw_survey(boundary=1e-4,
                                       flux_sigma=0.1,
                                       hard_cut=False,
                                       verbose=verbose)

        self.reset()

        return pop

    def draw_z_select(self, verbose):

        pop = self.pop_gen.draw_survey(boundary=1e-6,
                                       flux_sigma=0.5,
                                       distance_probability=0.5,
                                       verbose=verbose)

        self.reset()

        return pop

    def draw_hard_with_selector(self, verbose):

        selector = popsynth.HardFluxSelection(boundary=1e-4)

        self.pop_gen.set_flux_selection(selector)

        pop = self.pop_gen.draw_survey(boundary=1e-6,
                                       flux_sigma=0.5,
                                       verbose=verbose)

        self.reset()

        return pop

    def draw_soft_with_selector(self, verbose):

        selector = popsynth.SoftFluxSelection(boundary=1e-4, strength=10)

        self.pop_gen.set_flux_selection(selector)

        pop = self.pop_gen.draw_survey(boundary=1e-6,
                                       flux_sigma=0.5,
                                       verbose=verbose)

        self.reset()

        return pop

    def reset(self):

        self.pop_gen._distance_selector_set = False
        self.pop_gen._flux_selector_set = False
        self.pop_gen._flux_selector = popsynth.UnitySelection()
        self.pop_gen._distance_selector = popsynth.UnitySelection()

    def test_it(self):

        self.pop_gen.display()

        self.pop_gen.graph

        #####################

        pop = self.draw_hard(verbose=True)
        pop = self.draw_hard(verbose=False)

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")

        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        assert population_reloaded.hard_cut

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_soft(verbose=True)
        pop = self.draw_soft(verbose=False)

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

        assert not population_reloaded.hard_cut

        assert population_reloaded.distance_probability == 1.0

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_z_select(verbose=True)
        pop = self.draw_z_select(verbose=False)

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")
        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        assert population_reloaded.distance_probability < 1.0

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_all(verbose=True)
        pop = self.draw_all(verbose=False)

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

        pop = self.draw_none(verbose=True)
        pop = self.draw_none(verbose=False)

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
        assert population_reloaded.n_objects == population_reloaded.n_non_detections
        assert population_reloaded.n_detections == 0

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_hard_with_selector(verbose=True)
        pop = self.draw_hard_with_selector(verbose=False)

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")

        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        assert population_reloaded.boundary == 1e-4

        os.remove("_saved_pop.h5")

        #####################

        pop = self.draw_soft_with_selector(verbose=True)
        pop = self.draw_soft_with_selector(verbose=False)

        pop.display()

        fig = pop.display_fluxes()

        fig = pop.display_flux_sphere()

        pop.writeto("_saved_pop.h5")
        population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

        assert population_reloaded.boundary == 1e-4

        assert population_reloaded.distance_probability == 1.0

        os.remove("_saved_pop.h5")

        # clean up
        plt.close("all")


def test_spatial_population_with_derived():

    for pop, param in zip(_spatial_dict, _spatial_params):

        param = copy.deepcopy(param)

        pb = Popbuilder(pop, **param)

        # first make sure they all fail

        with pytest.raises(AssertionError):
            pb.draw_hard(verbose=True)
        with pytest.raises(AssertionError):
            pb.draw_soft(verbose=True)
        with pytest.raises(AssertionError):
            pb.draw_z_select(verbose=True)

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
