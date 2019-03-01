# Scientific libraries
import numpy as np
import scipy.stats as stats
import scipy.special as sf
import scipy.integrate as integrate
import pandas as pd
import abc
from IPython.display import display, Math, Markdown

from popsynth.population import Population
from popsynth.auxiliary_sampler import DerivedLumAuxSampler
#from popsynth.utils.progress_bar import progress_bar
from tqdm.autonotebook import tqdm as progress_bar

class PopulationSynth(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, r_max=10, seed=1234, name='no_name'):
        """FIXME! briefly describe function

        :param r_max: 
        :param seed: 
        :param name: 
        :returns: 
        :rtype: 

        """

        self._n_model = 500
        self._seed = int(seed)
        self._model_spaces = {}
        self._auxiliary_observations = {}
        self._name = name

        self._r_max = r_max

        self._has_derived_luminosity = False
        self._derived_luminosity_sampler = None

    def set_luminosity_function_parameters(self, **lf_params):
        """
        Set the luminosity function parameters as keywords
        """

        try:
            for k, v in lf_params.items():

                if k in self._lf_params:
                    self._lf_params[k] = v

                else:
                    RuntimeWarning('%s was not originally in the parameters... ignoring.' % k)

        except:

            # we have not set params before

            self._lf_params = lf_params

    def set_spatial_distribution_params(self, **spatial_params):
        """
        Set the spatial parameters as keywords
        """

        try:

            for k, v in spatial_params.items():

                if k in self._spatial_params:
                    self._spatial_params[k] = v
                else:
                    RuntimeWarning('%s was not originally in the parameters... ignoring.' % k)

        except:

            # we have not set these before

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
        """FIXME! briefly describe function

        :param auxiliary_sampler: 
        :returns: 
        :rtype: 

        """

        if isinstance(auxiliary_sampler, DerivedLumAuxSampler):
            self._has_derived_luminosity = True
            self._derived_luminosity_sampler = auxiliary_sampler

        else:

            self._auxiliary_observations[auxiliary_sampler.name] = auxiliary_sampler

    @property
    def name(self):
        return self._name

    # The following methods must be implemented in subclasses

    @abc.abstractmethod
    def phi(self, L):

        raise RuntimeError('Must be implemented in derived class')

        pass

    @abc.abstractmethod
    def differential_volume(self, distance):

        raise RuntimeError('Must be implemented in derived class')
        pass

    @abc.abstractmethod
    def dNdV(self, distance):

        raise RuntimeError('Must be implemented in derived class')
        pass

    def time_adjustment(self, r):
        """FIXME! briefly describe function

        :param r: 
        :returns: 
        :rtype: 

        """

        return 1.

    def draw_distance(self, size, verbose):
        """
        Draw the distances from the specified dN/dr model
        """

        # create a callback for the sampler
        dNdr = lambda r: self.dNdV(r) * self.differential_volume(r) / self.time_adjustment(r)

        # find the maximum point
        tmp = np.linspace(0, self._r_max, 500)
        ymax = np.max(dNdr(tmp))

        # rejection sampling the distribution
        r_out = []

        if verbose:
            for i in progress_bar(range(size), desc='Drawing distances'):
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
        else:

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
            

        return np.array(r_out)

    @abc.abstractmethod
    def draw_luminosity(self, size):
        pass

    @abc.abstractmethod
    def transform(self, flux, distance):
        pass

    def _prob_det(self, x, boundary, strength):
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

    def draw_survey(self, boundary, flux_sigma=1., strength=10., hard_cut=False, distance_probability=None, verbose=True):
        """
        Draw the total survey and return a Population object

        :param boundary: the mean boundary for flux selection
        :param flux_sigma: the homoskedastic sigma for the flux in log10 space
        :param strength: the log10 strength of the inv logit selection
        :param hard_cut: (bool) If true, had cuts are applid to the selection
        :param distance_probability: If not none, then the probability of detecting a distance
        :return: a Population object
        """

        # set the random seed

#        pbar = progress_bar(total=5, desc='Integrating volume')
        np.random.seed(self._seed)

        # create a callback of the integrand
        dNdr = lambda r: self.dNdV(r) * self.differential_volume(r) / self.time_adjustment(r)

        # integrate the population to determine the true number of
        # objects
        N = integrate.quad(dNdr, 0., self._r_max)[0]

        # this should be poisson distributed
        n = np.random.poisson(N)
 #       pbar.update()
 #       pbar.set_description(desc='Drawing distances')
        distances = self.draw_distance(size=n, verbose=verbose)

        if verbose:
            print('Expecting %d total objects' % n)

        # first check if the auxilliary samplers
        # compute the luminosities

        #      pbar.update()



        # now we set up the selection that _may_ come
        # from the auxilliary samplers

        auxiliary_selection = np.ones(n, dtype=bool)
        
  

        auxiliary_quantities = {}

        # this means the luminosity is not
        # simulated directy

        if self._has_derived_luminosity:

            #pbar.set_description(desc='Getting derived luminosities')
            # set the distance to the auxilary sampler
            self._derived_luminosity_sampler.set_distance(distances)

            # sample the true and obs
            # values which are held internally

            self._derived_luminosity_sampler.draw(size=n)

            # check to make sure we sampled!
            assert self._derived_luminosity_sampler.true_values is not None and len(
                self._derived_luminosity_sampler.true_values) == n

            assert self._derived_luminosity_sampler.obs_values is not None and len(
                self._derived_luminosity_sampler.obs_values) == n

            # append these values to a dict
            auxiliary_quantities[self._derived_luminosity_sampler.name] = {
                'true_values': self._derived_luminosity_sampler.true_values,
                'obs_values': self._derived_luminosity_sampler.obs_values,
                'sigma': self._derived_luminosity_sampler.sigma,
                'selection' : self._derived_luminosity_sampler.selection
            }

            print('Getting luminosity from derived sampler')
            luminosities = self._derived_luminosity_sampler.compute_luminosity()

            # collect anything that was sampled here

            for k2, v2 in self._derived_luminosity_sampler.secondary_samplers.items():
                
                # first we tell the sampler to go and retrieve all of
                # its own secondaries

                
                properties = v2.get_secondary_properties()

                for k3, v3 in properties.items():

                    # now attach them
                    auxiliary_quantities[k3] = v3
 
            #pbar.update()

        else:
            #pbar.update()
            # draw all the values
            luminosities = self.draw_luminosity(size=n)

        # transform the fluxes
        fluxes = self.transform(luminosities, distances)

        # now sample any auxilary quantities
        # if needed

        #pbar.set_description(desc='Drawing Auxiliary variables')
        for k, v in self._auxiliary_observations.items():

            assert not v.is_secondary, 'This is a secondary sampler. You cannot sample it in the main sampler'

            # set the luminosities and distances to
            # auxilary sampler just in case
            # they are needed

            v.set_luminosity(luminosities)
            v.set_distance(distances)

            # sample the true and obs
            # values which are held internally
            # this will also invoke secondary samplers

            v.draw(size=n)

            # check to make sure we sampled!
            assert v.true_values is not None and len(v.true_values) == n
            assert v.obs_values is not None and len(v.obs_values) == n

            # append these values to a dict
            auxiliary_quantities[k] = {'true_values': v.true_values, 'sigma': v.sigma, 'obs_values': v.obs_values, 'selection': v.selection}

            # collect the secondary values

            for k2, v2 in v.secondary_samplers.items():

                # first we tell the sampler to go and retrieve all of
                # its own secondaries

                properties = v2.get_secondary_properties()

                for k3, v3 in properties.items():

                    # now attach them
                    auxiliary_quantities[k3] = v3
                    
       # pbar.update()
        
        # now draw all the observed fluxes
        # this is homoskedastic for now
        log10_fluxes_obs = self.draw_log10_fobs(fluxes, flux_sigma, size=n)

        # compute the detection probability  for the observed values
        #pbar.set_description(desc='Selecting objects')
        detection_probability = self._prob_det(log10_fluxes_obs, np.log10(boundary), strength)

        # now select them

        if not hard_cut:

            selection = []
            for p in progress_bar(detection_probability, desc='samping detection probability'):

                # make a bernoulli draw given the detection probability
                if stats.bernoulli.rvs(p) == 1:

                    selection.append(True)

                else:

                    selection.append(False)

            selection = np.array(selection)

        else:

            # simply apply a hard cut selection in the data
            
            selection = np.power(10, log10_fluxes_obs) >= boundary


        # now apply the selection from the auxilary samplers

        for k,v in auxiliary_quantities.items():

            auxiliary_selection = np.logical_and(auxiliary_selection, v['selection'])

        selection = np.logical_and(selection, auxiliary_selection)

            
        #pbar.update()
        if sum(selection) == n:
            print('NO HIDDEN OBJECTS')

        
        if distance_probability is not None:
            #pbar.set_description(desc='Selecting sistances')
            known_distances = []
            known_distance_idx = []
            unknown_distance_idx = []

            assert (distance_probability >= 0) and (
                distance_probability <= 1.), 'the distance detection must be between 0 and 1'

            with progress_bar(len(distances[selection]), desc='Selecting distances') as pbar2:
                for i, d in enumerate(distances[selection]):

                    # see if we detect the distance
                    if stats.bernoulli.rvs(distance_probability) == 1:

                        known_distances.append(d)
                        known_distance_idx.append(i)

                    else:

                        unknown_distance_idx.append(i)

                    pbar2.update()
            print('Detected %d distances' % len(known_distances))

        else:

            known_distances = distances[selection]
            known_distance_idx = [i for i in range(sum(selection))]
            unknown_distance_idx = []
        #pbar.update()
        known_distances = np.array(known_distances)
        known_distance_idx = np.array(known_distance_idx)
        unknown_distance_idx = np.array(unknown_distance_idx)

        try:
            print('Deteced %d objects or to a distance of %.2f' % (sum(selection), max(known_distances)))

        except:
            print('No Objects detected')
        # return a Population object

        return Population(
            luminosities=luminosities,
            distances=distances,
            known_distances=known_distances,
            known_distance_idx=known_distance_idx,
            unknown_distance_idx=unknown_distance_idx,
            fluxes=fluxes,
            flux_obs=np.power(10, log10_fluxes_obs),
            selection=selection,
            flux_sigma=flux_sigma,
            r_max=self._r_max,
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
            auxiliary_quantities=auxiliary_quantities)

    def display(self):
        """
        Display the simulation parameters
        
        """

        out = {'parameter': [], 'value': []}

        display(Markdown('## Luminosity Function'))
        for k, v in self._lf_params.items():

            out['parameter'].append(k)
            out['value'].append(v)

        display(Math(self._lf_form))
        display(pd.DataFrame(out))
        out = {'parameter': [], 'value': []}

        display(Markdown('## Spatial Function'))

        for k, v in self._spatial_params.items():

            out['parameter'].append(k)
            out['value'].append(v)

        display(Math(self._spatial_form))
        display(pd.DataFrame(out))

    def generate_stan_code(self, stan_gen, **kwargs):

        pass
