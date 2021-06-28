import numpy as np
import scipy.stats as stats
from popsynth.distribution import LuminosityDistribution, DistributionParameter


class BPLDistribution(LuminosityDistribution):
    _distribution_name = "BPLDistribution"

    Lmin = DistributionParameter(vmin=0)
    alpha = DistributionParameter()
    Lbreak = DistributionParameter(vmin=0)
    beta = DistributionParameter()
    Lmax = DistributionParameter(vmin=0)

    def __init__(self, seed: int = 1234, name: str = "bpl"):
        """A broken power law luminosity distribution.

        L ~ L^``alpha`` for L <= ``Lbreak``
        L ~ L^``beta`` for L > ``Lbreak``

        :param seed: Random seed
        :type seed: int
        :param name: Name of the distribution
        :type name: str
        :param Lmin: Minimum value of the luminosity
        :type Lmin: :class:`DistributionParameter`
        :param alpha: Index of the lower power law
        :type alpha: :class:`DistributionParameter`
        :param Lbreak: Luminosity of the power law break
        :type Lbreak: :class:`DistributionParameter`
        :param beta: Index of the upper power law
        :type beta: :class:`DistributionParameter`
        :param Lmax: Maximum value of the luminosity
        :type Lmax: :class:`DistributionParameter`
        """
        lf_form = r"\begin{cases} C L^{\alpha} & \mbox{if } L"
        lf_form += r"\leq L_\mathrm{break},\\ C L^{\beta} "
        lf_form += r"L_\mathrm{break}^{\alpha - \beta}"
        lf_form += r" & \mbox{if } L > L_\mathrm{break}. \end{cases}"

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
    Integrate a broken power law between bounds.

    :param x0: Lower bound
    :param x1: Break point
    :param x2: Upper bound
    :param a1: Lower power law index
    :param a2: Upper power low index
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
    Broken power law between bounds.

    :param x: The domain of the function
    :param x0: Lower bound
    :param x1: Break point
    :param x2: Upper bound
    :param a1: Lower power law index
    :param a2: Upper power low index
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
    Sample from a broken power law
    between bounds.

    :param u: Uniform random number on {0,1}
    :param x0: Lower bound
    :param x1: Break point
    :param x2: Upper bound
    :param a1: Lower power law index
    :param a2: Upper power low index
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
