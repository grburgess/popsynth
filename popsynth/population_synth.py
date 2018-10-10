# Scientific libraries
import numpy as np
import scipy.stats as stats
import scipy.special as sf
import scipy.integrate as integrate

import abc
from IPython.display import display 

import h5py
import pandas as pd

from astropy.cosmology import WMAP9 as cosmo

from astropy.constants import c as sol
sol = sol.value

from popsynth.population import Population




class PopulationSynth(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, r_max=10, seed = 1234):
        self._n_model = 500
        self._seed int(seed)
        self._model_spaces = {}

 
        #self._lf_params = {}
        #self._spatial_params = {}

        self._r_max = r_max
        
    def set_luminosity_function_parameters(self, **lf_params):

        self._lf_params = lf_params

    def set_spatial_distribution_params(self, **spatial_params):

        self._spatial_params = spatial_params

    def add_model_space(self, name, start, stop, log=True):

        if log:
            space = np.logspace(np.log10(start), np.log10(stop), self._n_model)

        else:

            space = np.linspace(start, stop, self._n_model)

        self._model_spaces[name] = space

    @abc.abstractmethod
    def phi(self, L):
        pass
        
    @abc.abstractmethod
    def differential_volume(self, distance):
        pass

    @abc.abstractmethod
    def dNdV(self, distance):

        pass

    def time_adjustment(self, r):

        return 1.

    def draw_distance(self, size):

        dNdr = lambda r: self.dNdV(r) * self.differential_volume(r) / self.time_adjustment(r)

        tmp = np.linspace(0, self._r_max, 1E5)

        ymax = np.max(dNdr(tmp))

        r_out = []

        for i in range(size):
            flag = True
            while flag:

                y = np.random.uniform(low=0, high=ymax)
                r = np.random.uniform(low=0, high=self._r_max)

                if y < dNdr(r):
                    r_out.append(r)
                    flag = False
        return np.array(r_out)

    @abc.abstractmethod
    def draw_luminosity(self, size):
        pass

    @abc.abstractmethod
    def transform(self, flux, distance):
        pass

    def prob_det(self, x, boundary, strength):

        return sf.expit(strength * (x - boundary))

    def draw_log10_fobs(self, f, f_sigma, size=1):

        log10_f = np.log10(f)

        log10_fobs = log10_f + np.random.normal(loc=0, scale=f_sigma, size=size)

        return log10_fobs

    def draw_survey(self, boundary, flux_sigma=1., strength=10.):

        np.random.seed(self._seed)

        dNdr = lambda r: self.dNdV(r) * self.differential_volume(r) / self.time_adjustment(r)

        N = integrate.quad(dNdr, 0., self._r_max)[0]

        # this should be poisson distributed
        n = np.random.poisson(N)

        luminosities = self.draw_luminosity(size=n)
        distances = self.draw_distance(size=n)
        fluxes = self.transform(luminosities, distances)

        #log10_fluxes = np.log10(fluxes)

        log10_fluxes_obs = self.draw_log10_fobs(fluxes, flux_sigma, size=n)

        detection_probability = self.prob_det(log10_fluxes_obs, np.log10(boundary), strength)

        selection = []
        for p in detection_probability:

            if stats.bernoulli.rvs(p) == 1:

                selection.append(True)

            else:

                selection.append(False)

        selection = np.array(selection)



        return Population(
            luminosities=luminosities,
            distances=distances,
            fluxes=fluxes,
            flux_obs=np.power(10, log10_fluxes_obs),
            selection=selection,
            flux_sigma=flux_sigma,
            n_model=self._n_model,
            lf_params=self._lf_params,
            spatial_params=self._spatial_params,
            model_spaces=self._model_spaces,
            boundary=boundary,
            strength=strength,
            seed=self._seed
        )

    def display(self):

        spatial_df = pd.Series(self._spatial_params)
        lf_df = pd.Series(self._lf_params)

        print('Luminosity Function:')

        display(lf_df)

        print('Spatial Parameters:')

        display(spatial_df)
        
