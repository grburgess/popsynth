import numpy as np

from popsynth.population_synth import PopulationSynth


class SphericalPopulation(PopulationSynth):

    def __init__(self, r_max=10, seed=1234, name='_sphere'):

        PopulationSynth.__init__( self, r_max, seed, name)

    def differential_volume(self, r):

        return 4 * np.pi * r * r

    def transform(self, L, r):

        return L/(4. * np.pi * r * r)
    
class ConstantSphericalPopulation(SphericalPopulation):


    def __init__(self, Lambda=1. ,r_max = 10., seed=1234, name='_cons_sphere'):

        super(ConstantSphericalPopulation, self).__init__(r_max, seed, name)


        self.set_spatial_distribution_params(Lambda=Lambda)


        self._spatial_form = r'\Lambda'
        
    def __get_Lambda(self):
             """Calculates the 'Lambda' property."""
             return self._spatial_params['Lambda']

    def ___get_Lambda(self):
         """Indirect accessor for 'Lambda' property."""
         return self.__get_Lambda()

    def __set_Lambda(self, Lambda):
         """Sets the 'Lambda' property."""
         self.set_spatial_distribution_params(Lambda=Lambda)

    def ___set_Lambda(self, Lambda):
         """Indirect setter for 'Lambda' property."""
         self.__set_Lambda(Lambda)

    Lambda = property(___get_Lambda, ___set_Lambda,
                     doc="""Gets or sets the Lambda.""")
        
    
    def dNdV(self, distance):

        return self._spatial_params['Lambda']


