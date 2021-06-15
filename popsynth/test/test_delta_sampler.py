import numpy as np
from popsynth.aux_samplers.delta_aux_sampler import DeltaAuxSampler


def test_delta_sampler():

    N = 100

    delta_sampler = DeltaAuxSampler(name="delta")

    delta_sampler.xp = 1.0

    delta_sampler.true_sampler(N)

    assert len(delta_sampler._true_values) == N

    assert np.all(delta_sampler._true_values) == 1.0
