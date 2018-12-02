import numpy as np
import multiprocessing as mp


def sample_one(ymax, idx1,  xbreak,  idx2,  xmin,  xmax, i):
    
    np.random.seed(int((i+1)*1000))
    
    flag = True
    while(flag):
        y_guess = np.random.uniform(0, ymax)
        x_guess =  np.random.uniform(1, xmax/xmin)

        x_test = bpl( x_guess,  idx1,  xbreak/xmin,  idx2 )
        if y_guess <= x_test:
            flag = False
    return x_guess

def bpl( x,  idx1,  xbreak,  idx2 ):
    if x < xbreak:
        return pow(x/xbreak,-idx1)
    else:
        return pow(x/xbreak,-idx2)





from popsynth.population_synth import PopulationSynth


class BPLPopulation(PopulationSynth):

    def __init__(self, Lmin, alpha, Lbreak, beta, Lmax,  r_max=10, seed=1234, name='_pareto'):

        PopulationSynth.__init__(self, r_max, seed, name)

        self.set_luminosity_function_parameters(Lmin=Lmin, alpha=alpha, Lbreak=Lbreak, beta=beta, Lmax=Lmax)


        self._lf_form = r"\frac{\alpha L_{\rm min}^{\alpha}}{L^{\alpha+1}}"
    

    def phi(self, L):

        out = np.zeros_like(L)

        idx = L>=self.Lmin
        
        out[idx] =  self.alpha*self.Lmin**self.alpha / L[idx]**(self.alpha+1)

        return out
        
    def draw_luminosity(self, size=1):

        
        ymax = bpl(1., self.alpha,  self.Lbreak/self.Lmin,  self.beta )
    
        result_list = []
    
    
        def log_result(result):
            # This is called whenever foo_pool(i) returns a result.
            # result_list is modified only by the main process, not the pool workers.
            result_list.append(result)
                
        with mp.Pool(8) as pool:
    
            for i in range(size):
                pool.apply_async(sample_one, args=(ymax, self.alpha,  self.Lbreak,  self.beta,  self.Lmin,  self.Lmax, i), callback = log_result)
            pool.close()
            pool.join()
  

    
        return np.array(result_list) * self.Lmin
    
        
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


    def __get_Lmax(self):
             """Calculates the 'Lmax' property."""
             return self._lf_params['Lmax']

    def ___get_Lmax(self):
         """Indirect accessor for 'Lmax' property."""
         return self.__get_Lmax()

    def __set_Lmax(self, Lmax):
         """Sets the 'Lmax' property."""
         self.set_luminosity_function_parameters(alpha=self.alpha, Lmax=Lmax)

    def ___set_Lmax(self, Lmax):
         """Indirect setter for 'Lmax' property."""
         self.__set_Lmax(Lmax)

    Lmax = property(___get_Lmax, ___set_Lmax,
                     doc="""Gets or sets the Lmax.""")



    

    def __get_Lbreak(self):
        """Calculates the 'Lbreak' property."""
        return self._lf_params['Lbreak']

    def ___get_Lbreak(self):
         """Indirect accessor for 'Lbreak' property."""
         return self.__get_Lbreak()

    def __set_Lbreak(self, Lbreak):
         """Sets the 'Lbreak' property."""
         self.set_luminosity_function_parameters(alpha=self.alpha, Lbreak=Lbreak)

    def ___set_Lbreak(self, Lbreak):
         """Indirect setter for 'Lbreak' property."""
         self.__set_Lbreak(Lbreak)

    Lbreak = property(___get_Lbreak, ___set_Lbreak,
                     doc="""Gets or sets the Lbreak.""")


    
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


    def __get_beta(self):
             """Calculates the 'beta' property."""
             return self._lf_params['beta']

    def ___get_beta(self):
         """Indirect accessor for 'beta' property."""
         return self.__get_beta()

    def __set_beta(self, beta):
         """Sets the 'beta' property."""
         self.set_luminosity_function_parameters(beta=beta,Lmin=self.Lmin)

    def ___set_beta(self, beta):
         """Indirect setter for 'beta' property."""
         self.__set_beta(beta)

    beta = property(___get_beta, ___set_beta,
                     doc="""Gets or sets the beta.""")



