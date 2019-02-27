import popsynth





def test_basic_population():

    homo_pareto_synth = popsynth.synths.ParetoHomogeneousSphericalPopulation(Lambda=0.25, Lmin=1, alpha=2.)


    population = homo_pareto_synth.draw_survey(boundary=1E-2, strength=20, flux_sigma= 0.1)
