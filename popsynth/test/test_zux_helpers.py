import numpy as np
from popsynth.aux_samplers.lognormal_aux_sampler import LogNormalAuxSampler
from popsynth.aux_samplers.normal_aux_sampler import NormalAuxSampler
from popsynth.aux_samplers.trunc_normal_aux_sampler import TruncatedNormalAuxSampler

from hypothesis import given, settings
import hypothesis.strategies as st


@given(
    st.floats(min_value=0.01,),
    st.floats(min_value=0.01,),
    st.integers(min_value=2, max_value=1000),
)
def test_lognorm_sampler(mu, tau, size):

    sampler = LogNormalAuxSampler("test", mu, tau, observed=False)

    sampler.true_sampler(size)

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size

    sampler = LogNormalAuxSampler("test", mu, tau, sigma=1.0, observed=True)

    sampler.true_sampler(size)

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size


@given(
    st.floats(), st.floats(min_value=0.01,), st.integers(min_value=2, max_value=1000)
)
def test_norm_sampler(mu, tau, size):

    sampler = NormalAuxSampler("test", mu, tau, observed=False)

    sampler.true_sampler(size)

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size

    sampler = NormalAuxSampler("test", mu, tau, sigma=1.0, observed=True)

    sampler.true_sampler(size)

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size


@given(
    st.floats(), st.floats(min_value=0.01,), st.integers(min_value=2, max_value=1000)
)
def test_truncnorm_sampler(mu, tau, size):

    sampler = TruncatedNormalAuxSampler("test", -10, 10, mu, tau, observed=False)

    sampler.true_sampler(size)

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size

    sampler = TruncatedNormalAuxSampler(
        "test", 0, 10, mu, tau, sigma=1.0, observed=True
    )

    sampler.true_sampler(size)

    sampler.observation_sampler(size)

    assert len(sampler._true_values) == size
