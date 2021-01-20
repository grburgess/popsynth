import numpy as np
import scipy.stats as stats
from popsynth.distribution import LuminosityDistribution, DistributionParameter


class BPLDistribution(LuminosityDistribution):

    Lmin = DistributionParameter(vmin=0)
    alpha = DistributionParameter()
    Lbreak = DistributionParameter(vmin=0)
    beta = DistributionParameter()
    Lmax = DistributionParameter(vmin=0)

    def __init__(self, seed=1234, name="bpl"):

        lf_form = r"\frac{\alpha L_{\rm min}^{\alpha}}{L^{\alpha+1}}"

        super(BPLDistribution, self).__init__(seed=seed,
                                              name=name,
                                              form=lf_form)

    def phi(self, L):

        return bpl(L, self.Lmin, self.Lbreak, self.Lmax, self.alpha, self.beta)

    def draw_luminosity(self, size=1):

        u = np.atleast_1d(np.random.uniform(size=size))

        return sample_bpl(u, self.Lmin, self.Lbreak, self.Lmax, self.alpha,
                          self.beta)


def integrate_pl(x0, x1, x2, a1, a2):
    """
    x0: lower bound
    x1: break point
    x2: upper bound
    a1: lower power law index
    a2: upper power low index

    """

    # compute the integral of each piece analytically
    int_1 = (np.power(x1, a1 + 1.0) - np.power(x0, a1 + 1.0)) / (a1 + 1)
    int_2 = (np.power(x1, a1 - a2) *
             (np.power(x2, a2 + 1.0) - np.power(x1, a2 + 1.0)) / (a2 + 1))

    # compute the total integral
    total = int_1 + int_2

    # compute the weights of each piece of the function
    w1 = int_1 / total
    w2 = int_2 / total

    return w1, w2, total


def bpl(x, x0, x1, x2, a1, a2):
    """
    x: the domain of the function
    x0: lower bound
    x1: break point
    x2: upper bound
    a1: lower power law index
    a2: upper power low index

    """

    # creatre a holder for the values
    out = np.empty_like(x)

    # get the total integral to compute the normalization
    _, _, C = integrate_pl(x0, x1, x2, a1, a2)
    norm = 1.0 / C

    # create an index to select each piece of the function
    idx = x < x1

    # compute the lower power law
    out[idx] = np.power(x[idx], a1)

    # compute the upper power law
    out[~idx] = np.power(x[~idx], a2) * np.power(x1, a1 - a2)

    return out * norm


def sample_bpl(u, x0, x1, x2, a1, a2):
    """
    u: uniform random number between on {0,1}
    x0: lower bound
    x1: break point
    x2: upper bound
    a1: lower power law index
    a2: upper power low index
    """

    # compute the weights with our integral function
    w1, w2, _ = integrate_pl(x0, x1, x2, a1, a2)

    # create a holder array for our output
    out = np.empty_like(u)

    # compute the bernoulli trials for lower piece of the function
    # *if we wanted to do the upper part... we just reverse our index*
    # We also compute these to bools for numpy array selection
    idx = stats.bernoulli.rvs(w1, size=len(u)).astype(bool)

    # inverse transform sample the lower part for the "successes"
    out[idx] = np.power(
        u[idx] * (np.power(x1, a1 + 1.0) - np.power(x0, a1 + 1.0)) +
        np.power(x0, a1 + 1.0),
        1.0 / (1 + a1),
    )

    # inverse transform sample the upper part for the "failures"
    out[~idx] = np.power(
        u[~idx] * (np.power(x2, a2 + 1.0) - np.power(x1, a2 + 1.0)) +
        np.power(x1, a2 + 1.0),
        1.0 / (1 + a2),
    )

    return out
