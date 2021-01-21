import scipy.stats as stats

import popsynth.populations as populations
from popsynth.aux_samplers import *
from popsynth.auxiliary_sampler import (AuxiliaryParameter, AuxiliarySampler,
                                        DerivedLumAuxSampler,
                                        NonObservedAuxSampler)
from popsynth.distribution import (DistributionParameter,
                                   LuminosityDistribution, SpatialDistribution)
from popsynth.population import Population
from popsynth.population_synth import PopulationSynth
from popsynth.populations import *
from popsynth.selection_probability import (BernoulliSelection,
                                            HardFluxSelection,
                                            SoftFluxSelection, UnitySelection)
from popsynth.utils.configuration import popsynth_config
from popsynth.utils.cosmology import cosmology
from popsynth.utils.logging import (activate_warnings, show_progress,
                                    silence_progress, silence_warnings,
                                    update_logging_level)

__all__ = [
    "AuxiliarySampler",
    "DerivedLumAuxSampler",
    "populations",
    "Population",
    "PopulationSynth",
]

# chance = stats.bernoulli.rvs(0.3)

# if chance:

#     from IPython.display import display, YouTubeVideo

#     display(YouTubeVideo("FYJ1dbyDcrI"))

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
