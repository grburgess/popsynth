import numpy as np
from popsynth.aux_samplers.lognormal_aux_sampler import LogNormalAuxSampler
from popsynth.aux_samplers.normal_aux_sampler import NormalAuxSampler
from popsynth.aux_samplers.trunc_normal_aux_sampler import TruncatedNormalAuxSampler

from hypothesis import given, settings
import hypothesis.strategies as st


def test_constructor():

    sampler = TruncatedNormalAuxSampler("test", observed=False)

    sampler.mu = 1
    sampler.tau = 2
    sampler.lower = -5
    sampler.upper = 10
    sampler.sigma = 0.4

    sampler2 = TruncatedNormalAuxSampler("test2", observed=False)

    sampler2.mu = 0.51
    sampler2.tau = 2.6
    sampler2.lower = -1
    sampler2.upper = 100
    sampler2.sigma = 5.0


    print(sampler.truth)

    assert sampler.mu == 1
    assert sampler.tau == 2
    assert sampler.lower == -5
    assert sampler.upper == 10
    assert sampler.sigma == 0.4


@given(
    st.floats(min_value=0.01, ),
    st.floats(min_value=0.01, max_value=10.0),
    st.integers(min_value=2, max_value=1000),
)
def test_lognorm_sampler(mu, tau, size):

    sampler = LogNormalAuxSampler("test", observed=False)

    sampler.true_sampler(size)

    sampler.mu = mu
    sampler.tau = tau

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size

    sampler = LogNormalAuxSampler("test", observed=True)

    sampler.mu = mu
    sampler.tau = tau
    sampler.sigma = 1

    sampler.true_sampler(size)

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size


@given(
    st.floats(),
    st.floats(min_value=0.01, max_value=10.0),
    st.integers(min_value=2, max_value=1000),
)
def test_norm_sampler(mu, tau, size):

    sampler = NormalAuxSampler("test", observed=False)

    sampler.true_sampler(size)

    sampler.mu = mu
    sampler.tau = tau
    sampler.sigma = 1

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size

    sampler = NormalAuxSampler("test", observed=True)

    sampler.mu = mu
    sampler.tau = tau
    sampler.sigma = 1

    sampler.true_sampler(size)

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size


@given(
    st.floats(min_value=-10, max_value=10),
    st.floats(min_value=0.01, max_value=10.0),
    st.integers(min_value=2, max_value=1000),
)
def test_truncnorm_sampler(mu, tau, size):

    sampler = TruncatedNormalAuxSampler("test", observed=False)

    sampler.mu = mu
    sampler.tau = tau
    sampler.lower = -10
    sampler.upper = 10
    sampler.sigma = 1

    sampler.true_sampler(size)

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size

    sampler = TruncatedNormalAuxSampler("test", observed=True)

    sampler.mu = mu
    sampler.tau = tau
    sampler.lower = -10
    sampler.upper = 10
    sampler.sigma = 1

    sampler.true_sampler(size)

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size
