import numpy as np


def sample_theta_phi(size: int):
    """
    Sample ``size`` samples uniformly on the
    surface of the unit sphere.
    """

    theta = np.arccos(1 - 2 * np.random.uniform(0.0, 1.0, size=size))
    phi = np.random.uniform(0, 2 * np.pi, size=size)

    return theta, phi


def xyz(r, theta, phi):
    """
    Convert spherical coordinates to Cartesian.
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z
