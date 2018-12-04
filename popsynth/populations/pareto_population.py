import numpy as np

from popsynth.population_synth import PopulationSynth


class ParetoPopulation(PopulationSynth):

    def __init__(self, Lmin, alpha, r_max=10, seed=1234, name='_pareto'):
        """
        A Pareto luminosity function

        :param Lmin: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :param name: 
        :returns: 
        :rtype: 

        """
        

        PopulationSynth.__init__(self, r_max, seed, name)

        self.set_luminosity_function_parameters(Lmin=Lmin, alpha=alpha)

        self._lf_form = r"\frac{\alpha L_{\rm min}^{\alpha}}{L^{\alpha+1}}"

    def phi(self, L):
        """
        The luminosity function

        :param L: 
        :returns: 
        :rtype: 

        """
        

        out = np.zeros_like(L)

        idx = L >= self.Lmin

        out[idx] = self.alpha * self.Lmin**self.alpha / L[idx]**(self.alpha + 1)

        return out

    def draw_luminosity(self, size=1):
        """FIXME! briefly describe function

        :param size: 
        :returns: 
        :rtype: 

        """
        
        return (np.random.pareto(self._lf_params['alpha'], size) + 1) * self._lf_params['Lmin']

    def __get_Lmin(self):
        """Calculates the 'Lmin' property."""
        return self._lf_params['Lmin']

    def ___get_Lmin(self):
        """Indirect accessor for 'Lmin' property."""
        return self.__get_Lmin()

    def __set_Lmin(self, Lmin):
        """Sets the 'Lmin' property."""
        self.set_luminosity_function_parameters(alpha=self.alpha, Lmin=Lmin)

    def ___set_Lmin(self, Lmin):
        """Indirect setter for 'Lmin' property."""
        self.__set_Lmin(Lmin)

    Lmin = property(___get_Lmin, ___set_Lmin, doc="""Gets or sets the Lmin.""")

    def __get_alpha(self):
        """Calculates the 'alpha' property."""
        return self._lf_params['alpha']

    def ___get_alpha(self):
        """Indirect accessor for 'alpha' property."""
        return self.__get_alpha()

    def __set_alpha(self, alpha):
        """Sets the 'alpha' property."""
        self.set_luminosity_function_parameters(alpha=alpha, Lmin=self.Lmin)

    def ___set_alpha(self, alpha):
        """Indirect setter for 'alpha' property."""
        self.__set_alpha(alpha)

    alpha = property(___get_alpha, ___set_alpha, doc="""Gets or sets the alpha.""")

    def generate_stan_code(self, stan_gen, **kwargs):

        pass
