from numba import jit, njit, prange, float64
import numpy as np


@jit(parallel=True, forceobj=True)
def rejection_sample(size, ymax, xmax, func):
    """FIXME! briefly describe function

    :param size: 
    :param ymax: 
    :param xmax: 
    :param func: 
    :returns: 
    :rtype: 

    """

    r_out = []

    for i in prange(size):
        flag = True
        while flag:

            # get am rvs from 0 to the max of the function

            y = np.random.uniform(low=0, high=ymax)

            # get an rvs from 0 to the maximum distance

            r = np.random.uniform(low=0, high=xmax)

            # compare them

            if y < func(r):
                r_out.append(r)
                flag = False

    return r_out
