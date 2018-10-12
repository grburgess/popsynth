# Scientific libraries
import numpy as np
import scipy.stats as stats
import scipy.special as sf
import scipy.integrate as integrate

import abc
from IPython.display import display, Math, Markdown

import h5py
import pandas as pd

from astropy.cosmology import WMAP9 as cosmo

from astropy.constants import c as sol
sol = sol.value

from popsynth.population import Population
from popsynth.utils.progress_bar import progress_bar



class PopulationSynth(object):
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, r_max=10, seed = 1234, name='no_name'):
        """
        
        """
        self._n_model = 500
        self._seed = int(seed)
        self._model_spaces = {}
        self._auxiliary_observations = {}
        self._name = name

        self._r_max = r_max
        
    def set_luminosity_function_parameters(self, **lf_params):
        """
        Set the luminosity function parameters as keywords
        """

        self._lf_params = lf_params

    def set_spatial_distribution_params(self, **spatial_params):
        """
        Set the spatial parameters as keywords
        """
        self._spatial_params = spatial_params

    def add_model_space(self, name, start, stop, log=True):
        """
        Add a model space for stan generated quantities
    
        :param name: name that Stan will use
        :param start: start of the grid
        :param stop: stop of the grid
        :param log: use log10 or not

        """
        if log:
            space = np.logspace(np.log10(start), np.log10(stop), self._n_model)
 
        else:

            space = np.linspace(start, stop, self._n_model)

        self._model_spaces[name] = space

    def add_observed_quantity(self, auxiliary_sampler):


        self._auxiliary_observations[auxiliary_sampler.name] = auxiliary_sampler
        
        
        
    @property
    def name(self):
        return self._name

    # The following methods must be implemented in subclasses
    
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
        """
        Draw the distances from the specified dN/dr model
        """

        # create a callback for the sampler
        dNdr = lambda r: self.dNdV(r) * self.differential_volume(r) / self.time_adjustment(r)

        # find the maximum point
        tmp = np.linspace(0, self._r_max, 1E5)
        ymax = np.max(dNdr(tmp))

        # rejection sampling the distribution
        r_out = []
        with progress_bar(size, title='Drawing distances') as pbar:
            for i in range(size):
                flag = True
                while flag:

                    # get am rvs from 0 to the max of the function
                    
                    y = np.random.uniform(low=0, high=ymax)

                    # get an rvs from 0 to the maximum distance
                    
                    r = np.random.uniform(low=0, high=self._r_max)

                    # compare them
                    
                    if y < dNdr(r):
                        r_out.append(r)
                        flag = False
                pbar.increase()
                
        return np.array(r_out)

    @abc.abstractmethod
    def draw_luminosity(self, size):
        pass

    @abc.abstractmethod
    def transform(self, flux, distance):
        pass

    def prob_det(self, x, boundary, strength):
        """
        Soft detection threshold

        :param x: values to test
        :param boundary: mean value of the boundary
        :param strength: the strength of the threshold
        """


        return sf.expit(strength * (x - boundary))

    def draw_log10_fobs(self, f, f_sigma, size=1):
        """
        draw the log10 of the the fluxes
        """

        log10_f = np.log10(f)

        # sample from the log distribution to keep positive fluxes
        log10_fobs = log10_f + np.random.normal(loc=0, scale=f_sigma, size=size)

        return log10_fobs

    def draw_survey(self, boundary, flux_sigma=1., strength=10.):

        np.random.seed(self._seed)

        dNdr = lambda r: self.dNdV(r) * self.differential_volume(r) / self.time_adjustment(r)

        N = integrate.quad(dNdr, 0., self._r_max)[0]
        
        # this should be poisson distributed
        n = np.random.poisson(N)

        print('Expecting %d total objects'%n)

        luminosities = self.draw_luminosity(size=n)
        distances = self.draw_distance(size=n)
        fluxes = self.transform(luminosities, distances)


        # now sample any auxilary quantities
        # if needed
        
        auxiliary_quantities = {}

        for k,v in self._auxiliary_observations.items():

            print('Sampling: %s' %k)

            v.set_luminosity(luminosities)
            v.set_distance(distances)

            v.true_sampler(size=n)
            v.observation_sampler(size=n)

            # check to make sure we sampled!
            assert v.true_values is not None and len(v.true_values) == n
            assert v.obs_values is not None and len(v.obs_values) == n
            
            auxiliary_quantities[k] = {'true_values': v.true_values,
                                       'obs_values': v.obs_values,
                                       'sigma': v.sigma 


            }
            
            
            
        
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

        print('Deteced %d objects or to a distance of %.2f' %(sum(selection), max(distances[selection])))


        return Population(
            luminosities=luminosities,
            distances=distances,
            fluxes=fluxes,
            flux_obs=np.power(10, log10_fluxes_obs),
            selection=selection,
            flux_sigma=flux_sigma,
            r_max = self._r_max,
            n_model=self._n_model,
            lf_params=self._lf_params,
            spatial_params=self._spatial_params,
            model_spaces=self._model_spaces,
            boundary=boundary,
            strength=strength,
            seed=self._seed,
            name=self._name,
            spatial_form=self._spatial_form,
            lf_form=self._lf_form,
            auxiliary_quantities=auxiliary_quantities
        )

    def display(self):
        """
        Display the simulation parameters
        
        """

        out={'parameter':[], 'value':[]}

        display(Markdown('## Luminosity Function'))
        for k,v in self._lf_params.items():

             out['parameter'].append(k)
             out['value'].append(v)

        display(Math(self._lf_form))
        display(pd.DataFrame(out))
        out={'parameter':[], 'value':[]}

        display(Markdown('## Spatial Function'))


        for k,v in self._spatial_params.items():

            out['parameter'].append(k)
            out['value'].append(v)

        display(Math(self._spatial_form))
        display(pd.DataFrame(out))
