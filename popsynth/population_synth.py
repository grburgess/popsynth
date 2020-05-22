import numpy as np
import scipy.stats as stats
import scipy.special as sf
import scipy.integrate as integrate
import pandas as pd
import abc
from IPython.display import display, Math, Markdown

import networkx as nx


from popsynth.population import Population
from popsynth.auxiliary_sampler import DerivedLumAuxSampler
from popsynth.utils.rejection_sample import rejection_sample
from popsynth.distribution import LuminosityDistribution, SpatialDistribution


# from popsynth.utils.progress_bar import progress_bar
from tqdm.autonotebook import tqdm as progress_bar
from numba import jit, njit, prange, float64


class PopulationSynth(object, metaclass=abc.ABCMeta):

    def __init__(
        self,
        spatial_distribution,
        luminosity_distribution=None,
        seed=1234,
        verbose=True,
    ):
        """FIXME! briefly describe function

        :param spatial_distribution: 
        :param luminosity_distribution: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        self._n_model = 500
        self._seed = int(seed)
        self._model_spaces = {}
        self._auxiliary_observations = {}

        self._graph = nx.DiGraph()

        self._verbose = verbose

        self._name = "%s" % spatial_distribution.name
        if luminosity_distribution is not None:
            self._name = "%s_%s" % (self._name, luminosity_distribution.name)
            assert isinstance(
                luminosity_distribution, LuminosityDistribution
            ), "the luminosity_distribution is the wrong type"

        assert isinstance(
            spatial_distribution, SpatialDistribution
        ), "the spatial_distribution is the wrong type"

        self._spatial_distribution = spatial_distribution
        self._luminosity_distribution = luminosity_distribution

        self._has_derived_luminosity = False
        self._derived_luminosity_sampler = None

        self._params = {}

        # keep a list of parameters here for checking

        for k, v in self._spatial_distribution.params.items():

            self._params[k] = v

        if self._luminosity_distribution is not None:
            for k, v in self._luminosity_distribution.params.items():

                self._params[k] = v

        self._graph.add_node(self._spatial_distribution.name)

        # add the sky sampler

        
        
    @property
    def spatial_distribution(self):
        return self._spatial_distribution

    @property
    def luminosity_distribution(self):
        return self._luminosity_distribution

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
            if self._verbose:
                print(
                    "registering derived luminosity sampler: %s"
                    % auxiliary_sampler.name
                )

            self._has_derived_luminosity = True
            self._derived_luminosity_sampler = auxiliary_sampler

        else:

            assert not auxiliary_sampler.is_secondary, f'{auxiliary_sampler.name} is already set as a secondary sampler!'
            assert auxiliary_sampler.name not in self._auxiliary_observations, f'{auxiliary_sampler.name} is already registered!'
            if self._verbose:
                print("registering auxilary sampler: %s" % auxiliary_sampler.name)

            self._auxiliary_observations[auxiliary_sampler.name] = auxiliary_sampler

    def _prob_det(self, x, boundary, strength):
        """
        Soft detection threshold

        :param x: values to test
        :param boundary: mean value of the boundary
        :param strength: the strength of the threshold
        """

        return sf.expit(strength * (x - boundary))

    @property
    def name(self):
        return self._name

    def draw_log10_fobs(self, f, f_sigma, size=1):
        """
        draw the log10 of the the fluxes
        """

        log10_f = np.log10(f)

        # sample from the log distribution to keep positive fluxes
        log10_fobs = log10_f + np.random.normal(loc=0, scale=f_sigma, size=size)

        return log10_fobs

    def draw_survey(
        self,
        boundary,
        flux_sigma=1.0,
        strength=10.0,
        hard_cut=False,
        distance_probability=None,
        no_selection=False,
        verbose=True,
    ):
        """
        Draw the total survey and return a Population object

        :param boundary: the mean boundary for flux selection
        :param flux_sigma: the homoskedastic sigma for the flux in log10 space
        :param strength: the log10 strength of the inv logit selection
        :param hard_cut: (bool) If true, had cuts are applid to the selection
        :param distance_probability: If not none, then the probability of detecting a distance
        :return: a Population object
        """

        # this stores all the "true" population values from all the samplers
        truth = dict()

        # store the spatial distribution truths
        truth[self._spatial_distribution.name] = self._spatial_distribution.truth

        # set the random seed

        #        pbar = progress_bar(total=5, desc='Integrating volume')
        np.random.seed(self._seed)

        # create a callback of the integrand
        dNdr = (
            lambda r: self._spatial_distribution.dNdV(r)
            * self._spatial_distribution.differential_volume(r)
            / self._spatial_distribution.time_adjustment(r)
        )

        # integrate the population to determine the true number of
        # objects
        N = integrate.quad(dNdr, 0.0, self._spatial_distribution.r_max)[0]

        if verbose:
            print("The volume integral is %f" % N)

        # this should be poisson distributed
        n = np.random.poisson(N)

        
        self._spatial_distribution.draw_distance(size=n, verbose=verbose)

        # now draw the sky positions
        
        self._spatial_distribution.draw_sky_positions(size=n)

        distances = self._spatial_distribution.distances
        
        if verbose:
            print("Expecting %d total objects" % n)

        # first check if the auxilliary samplers
        # compute the luminosities

        #      pbar.update()

        # now we set up the selection that _may_ come
        # from the auxilliary samplers

        auxiliary_selection = np.ones(n, dtype=bool)

        auxiliary_quantities = {}

        # this means the luminosity is not
        # simulated directy

        if self.luminosity_distribution is None:

            assert self._has_derived_luminosity

        if self._has_derived_luminosity:

            # pbar.set_description(desc='Getting derived luminosities')
            # set the distance to the auxilary sampler
            self._derived_luminosity_sampler.set_distance(distances)

            # sample the true and obs
            # values which are held internally

            self._derived_luminosity_sampler.draw(size=n, verbose=verbose)

            # check to make sure we sampled!
            assert (
                self._derived_luminosity_sampler.true_values is not None
                and len(self._derived_luminosity_sampler.true_values) == n
            )

            assert (
                self._derived_luminosity_sampler.obs_values is not None
                and len(self._derived_luminosity_sampler.obs_values) == n
            )

            # append these values to a dict
            auxiliary_quantities[self._derived_luminosity_sampler.name] = {
                "true_values": self._derived_luminosity_sampler.true_values,
                "obs_values": self._derived_luminosity_sampler.obs_values,
                "selection": self._derived_luminosity_sampler.selection,
            }
            if verbose:
                print("Getting luminosity from derived sampler")
            luminosities = self._derived_luminosity_sampler.compute_luminosity()

            # collect anything that was sampled here

            # store the truth from the derived lum sampler

            truth[
                self._derived_luminosity_sampler.name
            ] = self._derived_luminosity_sampler.truth

            for k2, v2 in self._derived_luminosity_sampler.secondary_samplers.items():

                # first we tell the sampler to go and retrieve all of
                # its own secondaries
                
                properties = v2.get_secondary_properties()

                for k3, v3 in properties.items():

                    # now attach them
                    auxiliary_quantities[k3] = v3

                # store the secondary truths
                # this will _could_ be clobbered later
                # but that is ok

                truth[v2.name] = v2.truth

            # pbar.update()

        else:
            # pbar.update()
            # draw all the values
            luminosities = self.luminosity_distribution.draw_luminosity(size=n)

            # store the truths from the luminosity distribution
            truth[
                self.luminosity_distribution.name
            ] = self.luminosity_distribution.truth

        # transform the fluxes
        fluxes = self._spatial_distribution.transform(luminosities, distances)

        # now sample any auxilary quantities
        # if needed

        # pbar.set_description(desc='Drawing Auxiliary variables')
        for k, v in self._auxiliary_observations.items():

            assert (
                not v.is_secondary
            ), "This is a secondary sampler. You cannot sample it in the main sampler"

            # set the luminosities and distances to
            # auxilary sampler just in case
            # they are needed

            v.set_luminosity(luminosities)
            v.set_distance(distances)

            # sample the true and obs
            # values which are held internally
            # this will also invoke secondary samplers

            v.draw(size=n, verbose=verbose)

            # store the auxilliary truths
            truth[v.name] = v.truth

            # check to make sure we sampled!
            assert v.true_values is not None and len(v.true_values) == n
            assert v.obs_values is not None and len(v.obs_values) == n

            # append these values to a dict
            auxiliary_quantities[k] = {
                "true_values": v.true_values,
                "obs_values": v.obs_values,
                "selection": v.selection,
            }

            # collect the secondary values

            for k2, v2 in v.secondary_samplers.items():

                # first we tell the sampler to go and retrieve all of
                # its own secondaries

                properties = v2.get_secondary_properties()

                for k3, v3 in properties.items():

                    # now attach them
                    auxiliary_quantities[k3] = v3

                # store the secondary truths

                truth[v2.name] = v2.truth

        # pbar.update()

        # now draw all the observed fluxes
        # this is homoskedastic for now
        log10_fluxes_obs = self.draw_log10_fobs(fluxes, flux_sigma, size=n)

        assert np.alltrue(np.isfinite(log10_fluxes_obs))
        
        # now select them

        if not hard_cut:

            if verbose:

                print("Applying soft boundary")

            # compute the detection probability  for the observed values

            detection_probability = self._prob_det(
                log10_fluxes_obs, np.log10(boundary), strength
            )

            selection = []
            if verbose:
                for p in progress_bar(
                    detection_probability, desc="samping detection probability"
                ):

                    # make a bernoulli draw given the detection probability

                    if stats.bernoulli.rvs(p) == 1:

                        selection.append(True)

                    else:

                        selection.append(False)

            else:

                for p in detection_probability:

                    # make a bernoulli draw given the detection probability
                    if stats.bernoulli.rvs(p) == 1:

                        selection.append(True)

                    else:

                        selection.append(False)

            selection = np.array(selection)

        else:

            if verbose:

                print("Applying hard boundary")

            # simply apply a hard cut selection in the data

            selection = np.power(10, log10_fluxes_obs) >= boundary

        # now apply the selection from the auxilary samplers

        for k, v in auxiliary_quantities.items():

            auxiliary_selection = np.logical_and(auxiliary_selection, v["selection"])

            if verbose:

                if sum(~v["selection"]) > 0:

                    print(
                        "Applying selection from %s which selected %d of %d objects"
                        % (k, sum(v["selection"]), len(v["selection"]))
                    )

        if verbose:

            if sum(~auxiliary_selection) > 0:
                print(
                    "Before auxiliary selection there were %d objects selected"
                    % sum(selection)
                )

        selection = np.logical_and(selection, auxiliary_selection)

        # if we do not want to add a selection effect
        if no_selection:
            if self._verbose:
                print("No Selection! Added back all objects")

            selection = np.ones_like(selection, dtype=bool)

        # pbar.update()
        if sum(selection) == n:

            if verbose:
                print("NO HIDDEN OBJECTS")

        if (distance_probability is not None) or (distance_probability == 1.0):
            # pbar.set_description(desc='Selecting sistances')
            known_distances = []
            known_distance_idx = []
            unknown_distance_idx = []

            assert (distance_probability >= 0) and (
                distance_probability <= 1.0
            ), "the distance detection must be between 0 and 1"

            if verbose:
                with progress_bar(
                    len(distances[selection]), desc="Selecting distances"
                ) as pbar2:
                    for i, d in enumerate(distances[selection]):

                        # see if we detect the distance
                        if stats.bernoulli.rvs(distance_probability) == 1:

                            known_distances.append(d)
                            known_distance_idx.append(i)

                        else:

                            unknown_distance_idx.append(i)

                        pbar2.update()

            else:

                for i, d in enumerate(distances[selection]):

                    # see if we detect the distance
                    if stats.bernoulli.rvs(distance_probability) == 1:

                        known_distances.append(d)
                        known_distance_idx.append(i)

                    else:

                        unknown_distance_idx.append(i)

            if verbose:
                print("Detected %d distances" % len(known_distances))

        else:

            known_distances = distances[selection]
            known_distance_idx = [i for i in range(sum(selection))]
            unknown_distance_idx = []
        # pbar.update()
        known_distances = np.array(known_distances)
        known_distance_idx = np.array(known_distance_idx)
        unknown_distance_idx = np.array(unknown_distance_idx)

        if verbose:
            try:

                print(
                    "Deteced %d objects or to a distance of %.2f"
                    % (sum(selection), max(known_distances))
                )

            except:
                print("No Objects detected")
        # return a Population object

        ## just to make sure we do not do anything nutty
        lf_params = None
        lf_form = None
        if self._luminosity_distribution is not None:

            lf_params = self._luminosity_distribution.params
            lf_form = self._luminosity_distribution.form

        if distance_probability is None:
            distance_probability = 1.0

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
            r_max=self._spatial_distribution.r_max,
            n_model=self._n_model,
            lf_params=lf_params,
            spatial_params=self._spatial_distribution.params,
            model_spaces=self._model_spaces,
            boundary=boundary,
            strength=strength,
            seed=self._seed,
            name=self._name,
            spatial_form=self._spatial_distribution.form,
            lf_form=lf_form,
            auxiliary_quantities=auxiliary_quantities,
            truth=truth,
            hard_cut=hard_cut,
            distance_probability=distance_probability,
            graph=self.graph,
            theta = self._spatial_distribution.theta,
            phi = self._spatial_distribution.phi
        )

    def display(self):
        """
        Display the simulation parameters

        """

        if self._luminosity_distribution is not None:

            out = {"parameter": [], "value": []}

            display(Markdown("## Luminosity Function"))
            for k, v in self._luminosity_distribution.params.items():

                out["parameter"].append(k)
                out["value"].append(v)

            display(Math(self._luminosity_distribution.form))
            display(pd.DataFrame(out))
        out = {"parameter": [], "value": []}

        display(Markdown("## Spatial Function"))

        for k, v in self._spatial_distribution.params.items():

            out["parameter"].append(k)
            out["value"].append(v)

        display(Math(self._spatial_distribution.form))
        display(pd.DataFrame(out))

    # def generate_stan_code(self, stan_gen, **kwargs):

    #     pass

    @property
    def graph(self):

        self._build_graph()

        return self._graph

    def _build_graph(self):
        """
        builds the graph for all the samplers

        :returns: 
        :rtype: 

        """

        # first check out the luminosity sampler

        self._graph.add_node("obs_flux", observed=True)
        self._graph.add_edge(self._spatial_distribution.name, "obs_flux")
        if self._has_derived_luminosity:

            self._graph.add_node(self._derived_luminosity_sampler.name)

            self._graph.add_edge(self._derived_luminosity_sampler.name, "obs_flux")

            if self._derived_luminosity_sampler.uses_distance:

                self._graph.add_edge(
                    self._spatial_distribution.name,
                    self._derived_luminosity_sampler.name,
                )

            for k2, v2 in self._derived_luminosity_sampler.secondary_samplers.items():

                self._graph.add_node(k2)
                self._graph.add_edge(k2, self._derived_luminosity_sampler.name)

                # pass the graph and the primary

                _ = v2.get_secondary_properties(
                    graph=self._graph,
                    primary=k2,
                    spatial_distribution=self._spatial_distribution,
                )

        else:
            self._graph.add_edge(self._luminosity_distribution.name, "obs_flux")
        # now do the same fro everything else

        for k, v in self._auxiliary_observations.items():

            assert (
                not v.is_secondary
            ), "This is a secondary sampler. You cannot sample it in the main sampler"

            self._graph.add_node(k, observed=False)

            if v.observed:
                self._graph.add_node(v.obs_name, observed=False)
                self._graph.add_edge(k, v.obs_name)

                if v.uses_distance:

                    self._graph.add_edge(self._spatial_distribution.name, k)

            for k2, v2 in v.secondary_samplers.items():

                # first we tell the sampler to go and retrieve all of
                # its own secondaries

                self._graph.add_edge(k2, k)
                self._graph.add_node(k2, observed=False)

                if v2.uses_distance:

                    self._graph.add_edge(self._spatial_distribution.name, k2)

                if v2.observed:

                    self._graph.add_node(v2.obs_name, observed=True)
                    self._graph.add_edge(k2, v2.obs_name)

                _ = v2.get_secondary_properties(
                    graph=self._graph,
                    primary=k2,
                    spatial_distribution=self._spatial_distribution,
                )
