import numpy as np
import scipy.stats as stats
from popsynth.distribution import LuminosityDistribution


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
    int_2 = (
        np.power(x1, a1 - a2)
        * (np.power(x2, a2 + 1.0) - np.power(x1, a2 + 1.0))
        / (a2 + 1)
    )

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
        u[idx] * (np.power(x1, a1 + 1.0) - np.power(x0, a1 + 1.0))
        + np.power(x0, a1 + 1.0),
        1.0 / (1 + a1),
    )

    # inverse transform sample the upper part for the "failures"
    out[~idx] = np.power(
        u[~idx] * (np.power(x2, a2 + 1.0) - np.power(x1, a2 + 1.0))
        + np.power(x1, a2 + 1.0),
        1.0 / (1 + a2),
    )

    return out


class BPLDistribution(LuminosityDistribution):
    def __init__(self, Lmin, alpha, Lbreak, beta, Lmax, seed=1234, name="bpl"):
        """FIXME! briefly describe function

        :param Lmin: 
        :param alpha: 
        :param Lbreak: 
        :param beta: 
        :param Lmax: 
        :param seed: 
        :param name: 
        :returns: 
        :rtype: 

        """

        truth = dict(Lmin=Lmin, alpha=alpha, Lbreak=Lbreak, beta=beta, Lmax=Lmax)

        lf_form = r"\frac{\alpha L_{\rm min}^{\alpha}}{L^{\alpha+1}}"
        super(BPLDistribution, self).__init__(
            seed=seed, name=name, form=lf_form, truth=truth
        )

        self._construct_distribution_params(
            Lmin=Lmin, alpha=alpha, Lbreak=Lbreak, beta=beta, Lmax=Lmax
        )

    def phi(self, L):

        out = np.zeros_like(L)

        idx = L >= self.Lmin

        out[idx] = self.alpha * self.Lmin ** self.alpha / L[idx] ** (self.alpha + 1)

        return out

    def draw_luminosity(self, size=1):

        u = np.atleast_1d(np.random.uniform(size=size))

        return sample_bpl(u, self.Lmin, self.Lbreak, self.Lmax, self.alpha, self.beta)

    def __get_Lmin(self):
        """Calculates the 'Lmin' property."""
        return self._params["Lmin"]

    def ___get_Lmin(self):
        """Indirect accessor for 'Lmin' property."""
        return self.__get_Lmin()

    def __set_Lmin(self, Lmin):
        """Sets the 'Lmin' property."""
        self.set_distribution_params(alpha=self.alpha, Lmin=Lmin)

    def ___set_Lmin(self, Lmin):
        """Indirect setter for 'Lmin' property."""
        self.__set_Lmin(Lmin)

    Lmin = property(___get_Lmin, ___set_Lmin, doc="""Gets or sets the Lmin.""")

    def __get_Lmax(self):
        """Calculates the 'Lmax' property."""
        return self._params["Lmax"]

    def ___get_Lmax(self):
        """Indirect accessor for 'Lmax' property."""
        return self.__get_Lmax()

    def __set_Lmax(self, Lmax):
        """Sets the 'Lmax' property."""
        self.set_distribution_params(alpha=self.alpha, Lmax=Lmax)

    def ___set_Lmax(self, Lmax):
        """Indirect setter for 'Lmax' property."""
        self.__set_Lmax(Lmax)

    Lmax = property(___get_Lmax, ___set_Lmax, doc="""Gets or sets the Lmax.""")

    def __get_Lbreak(self):
        """Calculates the 'Lbreak' property."""
        return self._params["Lbreak"]

    def ___get_Lbreak(self):
        """Indirect accessor for 'Lbreak' property."""
        return self.__get_Lbreak()

    def __set_Lbreak(self, Lbreak):
        """Sets the 'Lbreak' property."""
        self.set_distribution_params(alpha=self.alpha, Lbreak=Lbreak)

    def ___set_Lbreak(self, Lbreak):
        """Indirect setter for 'Lbreak' property."""
        self.__set_Lbreak(Lbreak)

    Lbreak = property(___get_Lbreak, ___set_Lbreak, doc="""Gets or sets the Lbreak.""")

    def __get_alpha(self):
        """Calculates the 'alpha' property."""
        return self._params["alpha"]

    def ___get_alpha(self):
        """Indirect accessor for 'alpha' property."""
        return self.__get_alpha()

    def __set_alpha(self, alpha):
        """Sets the 'alpha' property."""
        self.set_distribution_params(alpha=alpha, Lmin=self.Lmin)

    def ___set_alpha(self, alpha):
        """Indirect setter for 'alpha' property."""
        self.__set_alpha(alpha)

    alpha = property(___get_alpha, ___set_alpha, doc="""Gets or sets the alpha.""")

    def __get_beta(self):
        """Calculates the 'beta' property."""
        return self._params["beta"]

    def ___get_beta(self):
        """Indirect accessor for 'beta' property."""
        return self.__get_beta()

    def __set_beta(self, beta):
        """Sets the 'beta' property."""
        self.set_distribution_params(beta=beta, Lmin=self.Lmin)

    def ___set_beta(self, beta):
        """Indirect setter for 'beta' property."""
        self.__set_beta(beta)

    beta = property(___get_beta, ___set_beta, doc="""Gets or sets the beta.""")
