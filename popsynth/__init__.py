from popsynth.population_synth import PopulationSynth
from popsynth.spherical_population import ConstantSphericalPopulation, SphericalPopulation
from popsynth.cosmological_population import CosmologicalPopulation, SFRPopulation
from popsynth.pareto_population import ParetoPopulation
from popsynth.schechter_population import SchechterPopulation

from popsynth.population import Population
from popsynth.auxiliary_sampler import AuxiliarySampler

from popsynth import synths
#from popsynth.synths import ParetoConstantSphericalPopulation, ParetoSFRPopulation, SchechterSFRPopulation





import numpy as np

chance = np.random.uniform(0,1,size=1)

if chance <= 0.8:

    from IPython.display import display, YouTubeVideo


    
    display(YouTubeVideo('NFTaiWInZ44'))
