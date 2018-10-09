# Scientific libraries
import numpy as np
import scipy.stats as stats
import scipy.special as sf
import scipy.integrate as integrate
import scipy.interpolate as interpolate

import h5py

from astropy.cosmology import WMAP9 as cosmo


from astropy.constants import c as sol
sol = sol.value


class PopulationSynth(object):

    def __init__(self):
        pass


    def set_luminosity_function_parameters(self,lf_params):

        self._lf_params = lf_params

    def set_spatial_distribution_params(self, spatial_params):

        self._spatial_params

    
    def comoving_volume(self,r):
        pass

    def dNdV(self):

        pass

    def draw_distance(self,size):

        pass
    
    def draw_luminosity(self,size):
        pass

    def transform(self,flux, distance):
        pass
    
    def prob_det(x, boundary, strength):

        return sf.expit(10.*(x-boundary))

    def draw_log10_fobs(self, f, f_sigma, size=1):

        log10_f = np.log10(f)

        log10_fobs = log10_f + np.random.normal(loc=0, scale=f_sigma, size=size)

    def draw_survey(self):

        N = integrate.quad(dNdz,0.,zmax,args=( r0, rise, decay , peak))[0]


        # this should be poisson distributed
        n = np.random.poisson(N)
        
        luminosities = self.draw_luminosity(size=n)
        distances = self.draw_distance(size=n)
        fluxes = self.transform(luminosities, distances)

        log10_fluxes = np.log10(fluxes)

        log10_fluxes_obs = self.draw_log10_fobs(fluxes, flux_sigma, size=n)

        detection_probability = self.prob_det(log10_fluxes_obs, np.log10(boundary), strength)
        
        selection = []
        for p in detection_probability:
        
        if stats.bernoulli.rvs(p) == 1:
            
            selection.append(True)
            
        else:
            
            selection.append(False)
        
        selection = np.array(selection)   

        n_model = 500

