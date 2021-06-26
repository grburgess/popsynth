# from numba import jit, njit, prange, float64
import numpy as np


# @jit(parallel=False, forceobj=True)
def rejection_sample(size, ymax, xmax, func):
    """
    Rejection sample ``func`` up to ``ymax`` and ``xmax``.

    :param size: Number of samples
    :param ymax: Maximum value of y
    :param xmax: Maximum value of x
    :param func: Function
    """

    r_out = np.zeros(size)

    for i in range(size):

        while True:

            # get am rvs from 0 to the max of the function

            y = np.random.uniform(low=0, high=ymax)

            # get an rvs from 0 to the maximum distance

            r = np.random.uniform(low=0, high=xmax)

            # compare them

            if y < func(r):
                r_out[i] = r
                break

    return r_out
