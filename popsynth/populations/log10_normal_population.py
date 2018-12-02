import numpy as np

from popsynth.population_synth import PopulationSynth




class Log10NormalPopulation(PopulationSynth):

    def __init__(self, mu, tau, r_max=10, seed=1234, name='_lognorm'):

        PopulationSynth.__init__(self, r_max, seed, name)

        self.set_luminosity_function_parameters(mu=mu, tau=tau)

        self._lf_form = r"\frac{\alpha L_{\rm min}^{\alpha}}{L^{\alpha+1}}"
    

    def phi(self, L):

        return (1./(self.tau * L * np.sqrt(2* np.pi))) * np.exp(-(np.log10(L) - self.mu)**2/(2*self.tau**2))

    def draw_luminosity(self, size=1):
    
        x = np.random.normal(loc=self.mu, scale=self.tau, size=size)

        return np.power(10., x)
    
    def __get_mu(self):
             """Calculates the 'mu' property."""
             return self._lf_params['mu']

    def ___get_mu(self):
         """Indirect accessor for 'mu' property."""
         return self.__get_mu()

    def __set_mu(self, mu):
         """Sets the 'mu' property."""
         self.set_luminosity_function_parameters(mu=mu, tau=self.tau)

    def ___set_mu(self, mu):
         """Indirect setter for 'mu' property."""
         self.__set_mu(mu)

    mu = property(___get_mu, ___set_mu,
                     doc="""Gets or sets the mu.""")

    def __get_tau(self):
             """Calculates the 'tau' property."""
             return self._lf_params['tau']

    def ___get_tau(self):
         """Indirect accessor for 'tau' property."""
         return self.__get_tau()

    def __set_tau(self, tau):
         """Sets the 'tau' property."""
         self.set_luminosity_function_parameters(mu=self.mu, tau=tau)

    def ___set_tau(self, tau):
         """Indirect setter for 'tau' property."""
         self.__set_tau(tau)

    tau = property(___get_tau, ___set_tau,
                     doc="""Gets or sets the tau.""")





