from dataclasses import dataclass
import numpy as np

from popsynth.selection_probability import SelectionProbabilty

@dataclass(frozen=True, repr=False)
class RecursiveSecondary:
    true_values: np.ndarray
    obs_values: np.ndarray
    selection: SelectionProbabilty
