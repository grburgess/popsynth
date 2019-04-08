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

    homo_sch_synth = popsynth.populations.SchechterHomogeneousSphericalPopulation(
        Lambda=0.1, Lmin=1, alpha=2.0
    )
    homo_sch_synth.display()
    population = homo_sch_synth.draw_survey(boundary=1e-2, strength=50, flux_sigma=0.1)
    population.display_fluxes()
    population.display_fluxes_sphere()

    population.writeto("saved_pop.h5")
    population_reloaded = popsynth.Population.from_file("saved_pop.h5")

    sfr_synth = popsynth.populations.ParetoSFRPopulation(
        r0=10.0, rise=0.1, decay=2.0, peak=5.0, Lmin=1e52, alpha=1.0, seed=123
    )


def test_auxiliary_sampler():

    sfr_synth = popsynth.populations.ParetoSFRPopulation(
        r0=10.0, rise=0.1, decay=2.0, peak=5.0, Lmin=1e52, alpha=1.0, seed=123
    )

    d = DemoSampler(5, 1, 0.1)

    d2 = DemoSampler2(0, 1, 0.1)

    d2.set_secondary_sampler(d)


    