from pathlib import Path
import os

import numpy as np
import pytest

import popsynth
from popsynth.utils.package_data import get_path_of_data_file

popsynth.debug_mode()



class DemoSampler(popsynth.AuxiliarySampler):
    _auxiliary_param_name = "DemoSampler"
    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=2)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)

    def __init__(self):

        super(DemoSampler, self).__init__("demo", observed=False)

    def true_sampler(self, size):

        self._true_values = np.random.normal(self.mu, self.tau, size=size)


class DemoSampler2(popsynth.DerivedLumAuxSampler):
    _auxiliary_param_name = "DemoSampler2"
    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=2)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)
    sigma = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)

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


def test_basic_population():

    homo_pareto_synth = popsynth.populations.ParetoHomogeneousSphericalPopulation(
        Lambda=0.25, Lmin=1, alpha=2.0)

    population = homo_pareto_synth.draw_survey()

    homo_pareto_synth.display()

    population.display()

    population.display_fluxes()
    population.display_flux_sphere()

    # Test the dist prob extremes

    flux_selector = popsynth.SoftFluxSelection(1e-2, 50)

    homo_pareto_synth.set_flux_selection(flux_selector)

    population = homo_pareto_synth.draw_survey(flux_sigma=1)

    ###


    b_selector = popsynth.BernoulliSelection(probability=.5)
    
    flux_selector = popsynth.SoftFluxSelection(1e-2, 20)

    homo_pareto_synth.set_flux_selection(flux_selector)

    homo_pareto_synth.set_distance_selection(b_selector)
    
    
    population = homo_pareto_synth.draw_survey(flux_sigma=.1)

    u_select = popsynth.UnitySelection()

    homo_pareto_synth.set_distance_selection(u_select)
    
    population = homo_pareto_synth.draw_survey( flux_sigma=0.1,
                                               )

    homo_sch_synth = popsynth.populations.SchechterHomogeneousSphericalPopulation(
        Lambda=0.1, Lmin=1, alpha=2.0)
    homo_sch_synth.display()


    homo_sch_synth.set_flux_selection(flux_selector)
    
    
    population = homo_sch_synth.draw_survey( flux_sigma=0.1)
    population.display_fluxes()
    population.display_flux_sphere()

    print(population.truth)

    homo_sch_synth.set_flux_selection(u_select)
    
    homo_sch_synth.draw_survey(
                               flux_sigma=0.1,
                               )

    population.writeto("_saved_pop.h5")
    population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

    os.remove("_saved_pop.h5")

    sfr_synth = popsynth.populations.ParetoSFRPopulation(r0=10.0,
                                                         rise=0.1,
                                                         decay=2.0,
                                                         peak=5.0,
                                                         Lmin=1e52,
                                                         alpha=1.0,
                                                         seed=123)


def test_auxiliary_sampler():

    sfr_synth = popsynth.populations.ParetoSFRPopulation(r0=10.0,
                                                         rise=0.1,
                                                         decay=2.0,
                                                         peak=5.0,
                                                         Lmin=1e52,
                                                         alpha=1.0,
                                                         seed=123)

    d = DemoSampler()

    d2 = DemoSampler2()

    d2.set_secondary_sampler(d)


    sfr_synth.draw_survey(.1)



def test_loading_from_file():

    p = get_path_of_data_file("pop.yml")
    
    ps = popsynth.PopulationSynth.from_file(p)

    assert ps.luminosity_distribution.Lmin == 1e51
    assert ps.luminosity_distribution.alpha == 2

    assert ps.spatial_distribution.Lambda == 0.5


    ps.draw_survey()
