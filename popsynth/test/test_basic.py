import popsynth
import pytest
import os
import numpy as np


class DemoSampler(popsynth.AuxiliarySampler):

    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=2)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)

    def __init__(self):

        super(DemoSampler, self).__init__("demo", observed=False)

    def true_sampler(self, size):

        self._true_values = np.random.normal(self.mu, self.tau, size=size)


class DemoSampler2(popsynth.DerivedLumAuxSampler):

    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=2)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)
    sigma = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)

    def __init__(self):

        super(DemoSampler2, self).__init__("demo2")

    def true_sampler(self, size):

        secondary = self._secondary_samplers["demo"]

        self._true_values = (
            (np.random.normal(self.mu, self.tau, size=size))
            + secondary.true_values
            - np.log10(1 + self.distance)
        )

    def observation_sampler(self, size):

        self._obs_values = self._true_values + np.random.normal(
            0, self.sigma, size=size
        )

    def compute_luminosity(self):

        secondary = self._secondary_samplers["demo"]

        return (10 ** (self._true_values + 54)) / secondary.true_values


def test_basic_population():

    homo_pareto_synth = popsynth.populations.ParetoHomogeneousSphericalPopulation(
        Lambda=0.25, Lmin=1, alpha=2.0
    )

    population = homo_pareto_synth.draw_survey(
        boundary=1e-2, strength=20, flux_sigma=0.1
    )

    homo_pareto_synth.display()

    population.display()

    population.display_fluxes()
    population.display_flux_sphere()

    # Test the dist prob extremes

    population = homo_pareto_synth.draw_survey(
        boundary=1e-2, strength=20, flux_sigma=0.1, distance_probability=0.5
    )

    population = homo_pareto_synth.draw_survey(
        boundary=1e-2, strength=20, flux_sigma=0.1, distance_probability=1.0
    )

    population = homo_pareto_synth.draw_survey(
        boundary=1e-2, strength=20, flux_sigma=0.1, distance_probability=0.0
    )

    homo_sch_synth = popsynth.populations.SchechterHomogeneousSphericalPopulation(
        Lambda=0.1, Lmin=1, alpha=2.0
    )
    homo_sch_synth.display()
    population = homo_sch_synth.draw_survey(boundary=1e-2, strength=50, flux_sigma=0.1)
    population.display_fluxes()
    population.display_flux_sphere()

    print(population.truth)

    homo_sch_synth.draw_survey(
        boundary=1e-2, strength=50, flux_sigma=0.1, no_selection=True
    )

    population.writeto("_saved_pop.h5")
    population_reloaded = popsynth.Population.from_file("_saved_pop.h5")

    os.remove("_saved_pop.h5")

    sfr_synth = popsynth.populations.ParetoSFRPopulation(
        r0=10.0, rise=0.1, decay=2.0, peak=5.0, Lmin=1e52, alpha=1.0, seed=123
    )


def test_auxiliary_sampler():

    sfr_synth = popsynth.populations.ParetoSFRPopulation(
        r0=10.0, rise=0.1, decay=2.0, peak=5.0, Lmin=1e52, alpha=1.0, seed=123
    )

    d = DemoSampler()

    d2 = DemoSampler2()

    d2.set_secondary_sampler(d)
