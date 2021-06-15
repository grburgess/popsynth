import numpy as np
import pytest

from popsynth.aux_samplers.delta_aux_sampler import DeltaAuxSampler


def test_delta_sampler():

    N = 1000

    delta_sampler = DeltaAuxSampler(name="delta", observed=True)

    delta_sampler.xp = 1.0

    delta_sampler.true_sampler(N)

    assert len(delta_sampler._true_values) == N

    assert np.all(delta_sampler._true_values) == 1.0

    delta_sampler.observation_sampler(N)

    assert pytest.approx(np.std(delta_sampler._obs_values), 0.1) == 1
