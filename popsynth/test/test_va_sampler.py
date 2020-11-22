import numpy as np
from popsynth.aux_samplers.viewing_angle_sampler import ViewingAngleSampler

from hypothesis import given, settings
import hypothesis.strategies as st


@given(
    st.floats(min_value=0.01, max_value=90.0), st.integers(min_value=1, max_value=1000)
)
def test_va_sampler(angle, size):

    va_sample = ViewingAngleSampler()

    va_sample.max_angle = angle

    va_sample.true_sampler(size)

    assert len(va_sample._true_values) == size
