import numpy as np


from astropy.cosmology import WMAP9 as cosmo

from popsynth.population_synth import PopulationSynth


from astropy.constants import c as sol
sol = sol.value


h0 = 69.3
dh = sol * 1.e-3 / h0
Om = 0.286
Om_reduced = ((1-Om)/Om)
Om_sqrt = np.sqrt(Om)
Ode = 1 - Om -(cosmo.Onu0 + cosmo.Ogamma0)


def Phi(x):    
    x2 = x*x
    x3 = x*x*x 
    top = 1. + 1.320*x + 0.441 *x2 + 0.02656*x3
    bottom = 1. + 1.392*x + 0.5121*x2 + 0.03944*x3
    return top/bottom

def xx(z): 
    return Om_reduced / np.power(1+z,3)


def luminosity_distance(z):
    x = xx(z)
    z1 = 1+z
    val = (2 * dh * z1 / Om_sqrt) *( Phi(xx(0)) - 1./(np.sqrt(z1)) * Phi(x)) *3.086E24 # in cm
    return val

def a(z):
    return np.sqrt( np.power(1+z,3.)*Om +Ode)

def comoving_transverse_distance(z):
    return luminosity_distance(z)/(1.+z)

def differential_comoving_volume(z):
    td = (comoving_transverse_distance(z)/3.086E24) 
    return (dh*td*td/a(z)) * 1E-9 # Gpc^3


class CosmologicalPopulation(PopulationSynth):

    def __init__(self, r_max=10, seed=1234, name='cosmo'):

        PopulationSynth.__init__(self, r_max, seed, name)

    def differential_volume(self, z):

        td = (comoving_transverse_distance(z)/3.086E24) 
        return (dh*td*td/a(z)) * 1E-9 # Gpc^3


    def transform(self, L, z):

        return L/(4. * np.pi * luminosity_distance(z)**2)

    def time_adjustment(self, z):
        return (1+z)

class SFRPopulation(CosmologicalPopulation):

    def __init__(self, r0, rise, decay, peak, r_max=10, seed=1234, name='_sfrcosmo'):

        CosmologicalPopulation.__init__(self, r_max, seed, name)

        self.set_spatial_distribution_params(r0=r0, rise=rise, decay=decay, peak=peak)

        self._spatial_form = r'\rho_0 \frac{1+r \cdot z}{1+ \left(z/p\right)^d}'


    def dNdV(self, z):
        top = 1. + self.rise * z
        bottom = 1+np.power(z/self.peak, self.decay)
    
        return self.r0 * top/bottom

        
    def __get_r0(self):
             """Calculates the 'r0' property."""
             return self._spatial_params['r0']

    def ___get_r0(self):
         """Indirect accessor for 'r0' property."""
         return self.__get_r0()

    def __set_r0(self, r0):
         """Sets the 'r0' property."""
         self.set_spatial_distribution_params(r0=r0, rise=self.rise, decay=self.decay, peak=self.peak)

    def ___set_r0(self, r0):
         """Indirect setter for 'r0' property."""
         self.__set_r0(r0)

    r0 = property(___get_r0, ___set_r0,
                     doc="""Gets or sets the r0.""")

    def __get_rise(self):
             """Calculates the 'rise' property."""
             return self._spatial_params['rise']

    def ___get_rise(self):
         """Indirect accessor for 'rise' property."""
         return self.__get_rise()

    def __set_rise(self, rise):
         """Sets the 'rise' property."""
         self.set_spatial_distribution_params(r0=self.r0, rise=rise, decay=self.decay, peak=self.peak)

    def ___set_rise(self, rise):
         """Indirect setter for 'rise' property."""
         self.__set_rise(rise)

    rise = property(___get_rise, ___set_rise,
                     doc="""Gets or sets the rise.""")

    
    def __get_decay(self):
             """Calculates the 'decay' property."""
             return self._spatial_params['decay']

    def ___get_decay(self):
         """Indirect accessor for 'decay' property."""
         return self.__get_decay()

    def __set_decay(self, decay):
         """Sets the 'decay' property."""
         self.set_spatial_distribution_params(r0=self.r0, rise=self.rise, decay=decay, peak=self.peak)

    def ___set_decay(self, decay):
         """Indirect setter for 'decay' property."""
         self.__set_decay(decay)

    decay = property(___get_decay, ___set_decay,
                     doc="""Gets or sets the decay.""")


    def __get_peak(self):
        """Calculates the 'peak' property."""
        return self._spatial_params['peak']

    def ___get_peak(self):
         """Indirect accessor for 'peak' property."""
         return self.__get_peak()

    def __set_peak(self, peak):
         """Sets the 'peak' property."""
         self.set_spatial_distribution_params(r0=self.r0, rise=self.rise, decay=self.decay, peak=peak)

    def ___set_peak(self, peak):
         """Indirect setter for 'peak' property."""
         self.__set_peak(peak)

    peak = property(___get_peak, ___set_peak,
                     doc="""Gets or sets the peak.""")

