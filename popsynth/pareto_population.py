import numpy as np

from popsynth.population_synth import PopulationSynth


class ParetoPopulation(PopulationSynth):

    def __init__(self, Lmin, alpha, r_max=10, seed=1234, name='_pareto'):

        PopulationSynth.__init__(self, r_max, seed, name)

        self.set_luminosity_function_parameters(Lmin=Lmin, alpha=alpha)

    

    def phi(self, L):
        return self.alpha*self.Lmin**self.alpha / L**(self.alpha+1)

    def draw_luminosity(self, size=1):
    
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

    Lmin = property(___get_Lmin, ___set_Lmin,
                     doc="""Gets or sets the Lmin.""")


    
    def __get_alpha(self):
             """Calculates the 'alpha' property."""
             return self._lf_params['alpha']

    def ___get_alpha(self):
         """Indirect accessor for 'alpha' property."""
         return self.__get_alpha()

    def __set_alpha(self, alpha):
         """Sets the 'alpha' property."""
         self.set_luminosity_function_parameters(alpha=alpha,Lmin=self.Lmin)

    def ___set_alpha(self, alpha):
         """Indirect setter for 'alpha' property."""
         self.__set_alpha(alpha)

    alpha = property(___get_alpha, ___set_alpha,
                     doc="""Gets or sets the alpha.""")



