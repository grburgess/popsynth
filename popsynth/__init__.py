from popsynth.population_synth import PopulationSynth
from popsynth.populations import *
from popsynth.aux_samplers import *

from popsynth.population import Population
from popsynth.auxiliary_sampler import AuxiliarySampler, DerivedLumAuxSampler

from popsynth import synths



import numpy as np

chance = np.random.uniform(0,1,size=1)

if chance <= 0.3:

    from IPython.display import display, YouTubeVideo


    
    display(YouTubeVideo('EwToQRXlFfc'))
