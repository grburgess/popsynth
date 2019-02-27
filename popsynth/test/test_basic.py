import popsynth





def test_basic_population():

    homo_pareto_synth = popsynth.synths.ParetoHomogeneousSphericalPopulation(Lambda=0.25, Lmin=1, alpha=2.)

    population = homo_pareto_synth.draw_survey(boundary=1E-2, strength=20, flux_sigma= 0.1)


    homo_pareto_synth.display()

    population.display()


    population.display_fluxes();
    population.display_flux_sphere();

    homo_sch_synth = popsynth.synths.SchechterHomogeneousSphericalPopulation(Lambda=0.1, Lmin=1, alpha=2.)
    homo_sch_synth.display()
    population = homo_sch_synth.draw_survey(boundary=1E-2, strength=50, flux_sigma= 0.1)
    population.display_fluxes();
    population.display_fluxes_sphere();



    sfr_synth = popsynth.synths.ParetoSFRPopulation(r0=10., rise=.1, decay=2., peak=5., Lmin=1E52, alpha=1.,seed=123)
