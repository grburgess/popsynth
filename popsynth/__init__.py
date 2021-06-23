import popsynth.populations as populations
from popsynth.aux_samplers import *
from popsynth.auxiliary_sampler import (AuxiliaryParameter, AuxiliarySampler,
                                        DerivedLumAuxSampler,
                                        NonObservedAuxSampler)
from popsynth.distribution import (DistributionParameter,
                                   LuminosityDistribution, SpatialDistribution)
from popsynth.distributions import *
from popsynth.population import Population
from popsynth.population_synth import PopulationSynth
from popsynth.populations import *
from popsynth.selection_probability import *
from popsynth.utils.configuration import popsynth_config
from popsynth.utils.cosmology import cosmology
from popsynth.utils.logging import (activate_logs, activate_warnings,
                                    debug_mode, loud_mode, quiet_mode,
                                    show_progress_bars, silence_logs,
                                    silence_progress_bars, silence_warnings,
                                    update_logging_level)
from popsynth.utils.registry import (list_available_auxiliary_samplers,
                                     list_available_distributions,
                                     list_available_selection_functions)

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
