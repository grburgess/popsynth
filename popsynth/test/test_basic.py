import os
from pathlib import Path

import numpy as np
import pytest

import popsynth
from popsynth.utils.package_data import get_path_of_data_file

popsynth.debug_mode()


class DemoSampler(popsynth.AuxiliarySampler):
    _auxiliary_sampler_name = "DemoSampler"
    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=2)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)

    def __init__(self):

        super(DemoSampler, self).__init__("demo", observed=False)

    def true_sampler(self, size):

        self._true_values = np.random.normal(self.mu, self.tau, size=size)


class DemoSampler2(popsynth.DerivedLumAuxSampler):
    _auxiliary_sampler_name = "DemoSampler2"
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

    homo_pareto_synth = (
        popsynth.populations.ParetoHomogeneousSphericalPopulation(Lambda=0.25,
                                                                  Lmin=1,
                                                                  alpha=2.0))

    population = homo_pareto_synth.draw_survey()

    homo_pareto_synth.display()

    population.display()

    print(homo_pareto_synth)

    population.display_fluxes()
    population.display_flux_sphere()

    # Test the dist prob extremes

    flux_selector = popsynth.SoftFluxSelection()
    flux_selector.boundary = 1e-2
    flux_selector.strength = 50

    homo_pareto_synth.clean(reset=True)

    homo_pareto_synth.set_flux_selection(flux_selector)

    population = homo_pareto_synth.draw_survey(flux_sigma=1)

    ###

    homo_pareto_synth.clean(reset=True)
    b_selector = popsynth.BernoulliSelection()
    b_selector.probability = 0.5

    homo_pareto_synth.set_flux_selection(flux_selector)

    homo_pareto_synth.set_distance_selection(b_selector)

    population = homo_pareto_synth.draw_survey(flux_sigma=0.1)

    homo_pareto_synth.clean(reset=True)

    u_select = popsynth.UnitySelection()

    homo_pareto_synth.set_distance_selection(u_select)

    population = homo_pareto_synth.draw_survey(flux_sigma=0.1, )
    homo_pareto_synth.clean(reset=True)

    homo_sch_synth = (
        popsynth.populations.SchechterHomogeneousSphericalPopulation(
            Lambda=0.1, Lmin=1, alpha=2.0))
    homo_sch_synth.display()

    flux_selector = popsynth.HardFluxSelection()
    flux_selector.boundary = 1e-5

    homo_sch_synth.set_flux_selection(flux_selector)

    population = homo_sch_synth.draw_survey(flux_sigma=0.1)

    population.display_fluxes()
    population.display_flux_sphere()

    print(population.truth)

    homo_sch_synth.clean(reset=True)

    homo_sch_synth.set_flux_selection(u_select)

    homo_sch_synth.draw_survey(flux_sigma=0.1, )

    population.writeto("_saved_pop.h5")
    population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

    os.remove("_saved_pop.h5")

    sfr_synth = popsynth.populations.ParetoSFRPopulation(
        r0=10.0,
        rise=0.1,
        a=0.015,
        decay=2.0,
        peak=5.0,
        Lmin=1e52,
        alpha=1.0,
        seed=123,
    )


def test_auxiliary_sampler():

    sfr_synth = popsynth.populations.ParetoSFRPopulation(
        r0=10.0,
        rise=0.1,
        a=0.015,
        decay=2.0,
        peak=5.0,
        Lmin=1e52,
        alpha=1.0,
        seed=123,
    )

    d = DemoSampler()

    sfr_synth.add_observed_quantity(d)

    with pytest.raises(RuntimeError):

        sfr_synth.add_observed_quantity(d)

    with pytest.raises(RuntimeError):

        sfr_synth.add_observed_quantity(1)

    dic = sfr_synth.to_dict()

    sfr_synth.draw_survey(0.1)

    _ = popsynth.PopulationSynth.from_dict(dic)

    sfr_synth = popsynth.populations.ParetoSFRPopulation(
        r0=10.0,
        rise=0.1,
        a=0.015,
        decay=2.0,
        peak=5.0,
        Lmin=1e52,
        alpha=1.0,
        seed=123,
    )

    d = DemoSampler()

    d2 = DemoSampler2()

    d2.set_secondary_sampler(d)

    dic = sfr_synth.to_dict()

    _ = popsynth.PopulationSynth.from_dict(dic)

    sfr_synth.add_auxiliary_sampler(d2)

    with pytest.raises(RuntimeError):

        sfr_synth.add_auxiliary_sampler(d)

    sfr_synth.draw_survey(0.1)


def test_loading_from_file():

    p = get_path_of_data_file("pop.yml")

    ps = popsynth.PopulationSynth.from_file(p)

    assert ps.luminosity_distribution.Lmin == 1e51
    assert ps.luminosity_distribution.alpha == 2

    assert ps.spatial_distribution.Lambda == 0.5

    pop = ps.draw_survey()

    ps.write_to("/tmp/test.yml")

    pop.writeto("/tmp/_pop.h5")

    ####

    p = get_path_of_data_file("test_pop.yml")

    ps = popsynth.PopulationSynth.from_file(p)

    assert ps._auxiliary_observations["two"].name == "two"

    assert (ps._auxiliary_observations["two"].secondary_samplers["one"].name ==
            "one")


def test_errors():

    pg = popsynth.populations.SchechterHomogeneousSphericalPopulation(
        10, 1, 1.5)

    with pytest.raises(RuntimeError):

        pg.set_distance_selection(1)

    with pytest.raises(RuntimeError):

        pg.set_flux_selection(1)

    with pytest.raises(RuntimeError):

        pg.add_spatial_selector(1)

    sd = (popsynth.populations.spatial_populations.
          ConstantSphericalDistribution())

    with pytest.raises(RuntimeError):

        _ = popsynth.PopulationSynth(spatial_distribution=sd,
                                     luminosity_distribution=1)

    with pytest.raises(RuntimeError):

        _ = popsynth.PopulationSynth(spatial_distribution=1)

    pg = popsynth.PopulationSynth(spatial_distribution=sd)

    with pytest.raises(RuntimeError):

        pg.draw_survey()
