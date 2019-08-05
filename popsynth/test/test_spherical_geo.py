import numpy as np
from popsynth.utils.spherical_geometry import sample_theta_phi, xyz

from hypothesis import given, settings
import hypothesis.strategies as st




@given(st.floats(min_value=.1 ), st.floats(), st.floats())
def test_xyz(r, theta, phi):

    x,y,z = xyz(r, theta, phi)

@given(st.integers(min_value=1, max_value=1000))
def test_sample_theta(size):

    theta, phi = sample_theta_phi(size)

    assert len(theta) == size
    assert len(phi) == size
