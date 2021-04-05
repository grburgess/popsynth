from abc import ABCMeta, abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy.integrate as integrate
import scipy.special as sf
import scipy.stats as stats
import yaml
from IPython.display import Markdown, Math, display
# from numpy.typing import np.ndarray
from numba import float64, jit, njit, prange

from popsynth.auxiliary_sampler import AuxiliarySampler, DerivedLumAuxSampler
from popsynth.distribution import LuminosityDistribution, SpatialDistribution
from popsynth.population import Population
from popsynth.selection_probability import (BernoulliSelection,
                                            HardFluxSelection,
                                            SelectionProbabilty,
                                            SoftFluxSelection, UnitySelection)
from popsynth.utils.logging import setup_logger
from popsynth.utils.progress_bar import progress_bar
from popsynth.utils.registry import (auxiliary_parameter_registry,
                                     distribution_registry, selection_registry)

log = setup_logger(__name__)


class PopulationSynth(object, metaclass=ABCMeta):
    def __init__(
        self,
        spatial_distribution: SpatialDistribution,
        luminosity_distribution: Optional[LuminosityDistribution] = None,
        seed: int = 1234,
    ):
        """
        Basic and generic population synth. One specifies the spatial and luminosity distribution OR
        derived luminosity distribution and everything is setup.

        :param spatial_distribution:
        :param luminosity_distribution:
        :param seed:
        :returns:
        :rtype:

        """

        self._n_model = 500  # type: int
        self._seed = int(seed)  # type: int

        # this is a container for things passed to
        # stan
        self._model_spaces = {}  # type: dict

        self._auxiliary_observations = {}  # type: dict

        self._graph = nx.DiGraph()  # type: nx.Digraph

        self._name = "%s" % spatial_distribution.name  # type: str
        if luminosity_distribution is not None:
            self._name = "%s_%s" % (self._name, luminosity_distribution.name)
            assert isinstance(
                luminosity_distribution, LuminosityDistribution
            ), "the luminosity_distribution is the wrong type"

        assert isinstance(
            spatial_distribution,
            SpatialDistribution), "the spatial_distribution is the wrong type"

        self._spatial_distribution = spatial_distribution  # type: SpatialDistribution
        self._luminosity_distribution = (
            luminosity_distribution
        )  # type: Union[LuminosityDistribution, None]

        self._has_derived_luminosity = False  # type: bool
        self._derived_luminosity_sampler = (
            None)  # type: Union[DerivedLumAuxSampler, None]

        # set the selections be fully seen unless it is set by the user
        self._distance_selector: SelectionProbabilty = UnitySelection()
        self._flux_selector: SelectionProbabilty = UnitySelection()

        # check to see if the selectors are set
        self._distance_selector_set: bool = False
        self._flux_selector_set: bool = False
        self._spatial_selector: Union[SelectionProbabilty, None] = None

        self._params = {}  # type: dict

        # keep a list of parameters here for checking

        for k, v in self._spatial_distribution.params.items():

            self._params[k] = v

        if self._luminosity_distribution is not None:
            for k, v in self._luminosity_distribution.params.items():

                self._params[k] = v

        self._graph.add_node(self._spatial_distribution.name)

        # add the sky sampler

    @classmethod
    def from_file(self, file_name: str) -> "PopulationSynth":

        with open(file_name) as f:

            input: Dict[str, Any] = yaml.load(f, Loader=yaml.SafeLoader)

        if "luminosity distribution" in input:

            # try:

            # load the name and parameters

            tmp = input["luminosity distribution"]

            ld_name = list(tmp.keys())[0]

            # create the instance

            luminosity_distribtuion: LuminosityDistribution = distribution_registry[ld_name]

            # now set the values of the parameters

            log.debug(f"setting parameters for {ld_name}")

            for k, v in tmp[ld_name].items():

                log.debug(f"trying to set {k} to {v}")

                if k in luminosity_distribtuion.__class__.__dict__:

                    setattr(luminosity_distribtuion, k, float(v))

                log.debug(f"{luminosity_distribtuion.params}")

        else:

            luminosity_distribtuion = None

        # try

        tmp = input["spatial distribution"]

        sd_name = list(tmp.keys())[0]

        spatial_distribtuion: SpatialDistribution = distribution_registry[sd_name]

        for k, v in tmp[sd_name].items():

            log.debug(f"trying to set {k} to {v}")

            if k in spatial_distribtuion.__class__.__dict__:

                setattr(spatial_distribtuion, k, float(v))

        seed: int = input["seed"]

        # create the poopulation synth

        pop_synth: PopulationSynth = PopulationSynth(
            spatial_distribtuion, luminosity_distribution=luminosity_distribtuion, seed=seed)

        # if there is a flux selection
        # then add it on

        if "flux selection" in input:

            tmp = input["flux selection"]

            if tmp is not None:

                fs_name = list(tmp.keys())[0]

                # extract the parameters

                params = tmp[fs_name]

                if params is None:

                    params = {}

                log.debug(f"flux selection parameters {params}")

                # make sure they are all floats

                for k, v in params.items():

                    params[k] = float(v)

                fs = selection_registry.get(fs_name, **params)

                pop_synth.set_flux_selection(fs)

        if "distance selection" in input:

            tmp = input["distance selection"]

            if tmp is not None:

                ds_name = list(tmp.keys())[0]

                log.debug(f"adding distance selection {ds_name}")

                # extract the parameters

                params = tmp[ds_name]

                if params is None:

                    params = {}

                log.debug(f"distance selection parameters {params}")

                # make sure they are all floats

                for k, v in params.items():

                    params[k] = float(v)

                ds = selection_registry.get(ds_name, **params)

                pop_synth.set_distance_selection(ds)

        if "spatial selection" in input:

            tmp = input["spatial selection"]

            if tmp is not None:

                ss_name = list(tmp.keys())[0]

                log.debug(f"adding distance selection {ss_name}")

                # extract the parameters

                params = tmp[ss_name]

                if params is None:

                    params = {}

                log.debug(f"spatial selection parameters {params}")

                # make sure they are all floats

                for k, v in params.items():

                    params[k] = float(v)

                ss = selection_registry.get(ss_name, **params)

                pop_synth.add_spatial_selector(ss)

        # Now collect the auxiliary samplers

        # we will gather them so that we can sort out dependencies
        aux_samplers: Dict[str, AuxiliarySampler] = OrderedDict()
        secondary_samplers: [str, str] = OrderedDict()

        if "auxiliary samplers" in input:

            for k, v in input["auxiliary samplers"].items():

                # first we extract the required info

                sampler_name = k
                obj_name = v.pop("name")
                is_observed = v.pop("observed")

                # now we extract the selection
                # and secondary if they are there

                if "selection" in v:
                    selection = v.pop("selection")

                else:

                    selection = None

                if "secondary" in v:
                    secondary = v.pop("secondary")

                else:

                    secondary = None

                if "init variables" in v:
                    init_variables = v.pop("init variables")

                    if init_variables is None:

                        init_variables = {}

                else:

                    init_variables = {}

                # since we have popped everything
                # all that is left should be parameters

                params = v

                # now build the object

                try:

                    tmp: AuxiliarySampler = auxiliary_parameter_registry.get(
                        sampler_name,
                        name=obj_name,
                        observed=is_observed,

                        **init_variables)

                except(TypeError):

                    # try without name

                    try:

                        tmp: AuxiliarySampler = auxiliary_parameter_registry.get(
                            sampler_name,
                            # name=obj_name,
                            observed=is_observed,

                            **init_variables)

                    except(TypeError):

                        try:

                            tmp: AuxiliarySampler = auxiliary_parameter_registry.get(
                                sampler_name,
                                name=obj_name,
                                # observed=is_observed,

                                **init_variables)

                        except(TypeError):

                            tmp: AuxiliarySampler = auxiliary_parameter_registry.get(
                                sampler_name, **init_variables)

                            # we will do a dirty trick
                            tmp._name = obj_name
                            tmp._is_observed = is_observed

                log.debug(f"setting parameters for {sampler_name}: {obj_name}")

                # now set the parameters

                for k, v in params.items():

                    log.debug(f"trying to set {k} to {v}")

                    if k in tmp.__class__.__dict__:

                        setattr(tmp, k, float(v))

                    log.debug(f"{tmp.truth}")

                if selection is not None:

                    sel_name = list(selection.keys())[0]

                    log.debug(f"adding selection {sel_name}")

                    # extract the parameters

                    params = selection[sel_name]

                    if params is None:

                        params = {}

                    log.debug(f"selection parameters {params}")

                    # make sure they are all floats

                    for k, v in params.items():

                        params[k] = float(v)

                    selector = selection_registry.get(sel_name, **params)

                    tmp.set_selection_probability(selector)

                # now we store this sampler

                log.debug(f"{obj_name} built")

                aux_samplers[obj_name] = tmp

                # if there is a secondary sampler,
                # we need to make a mapping

                if secondary is not None:

                    secondary_samplers[obj_name] = secondary

        # Now we have collected all of the auxiliary samplers
        # we need to assign those which are secondary

        log.debug(f"have {list(aux_samplers.keys())} as aux samplers")
        
        for primary, secondary in secondary_samplers.items():

            # assign it
            aux_samplers[primary].set_secondary_sampler(
                aux_samplers[secondary])

        # now we need to pop all the secondaries from the main
        # list so that we do not double add

        for secondary in list(secondary_samplers.values()):

            if secondary in aux_samplers:

                aux_samplers.pop(secondary)

        # now we add the observed quantites on

        for k, v in aux_samplers.items():

            pop_synth.add_observed_quantity(v)

        return pop_synth

    @property
    def spatial_distribution(self) -> SpatialDistribution:
        return self._spatial_distribution

    @property
    def luminosity_distribution(self) -> Union[LuminosityDistribution, None]:
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

    def add_observed_quantity(self,
                              auxiliary_sampler: Union[DerivedLumAuxSampler,
                                                       AuxiliarySampler]):
        """FIXME! briefly describe function

        :param auxiliary_sampler:
        :returns:
        :rtype:

        """

        if isinstance(auxiliary_sampler, DerivedLumAuxSampler):

            log.info("registering derived luminosity sampler: %s" %
                     auxiliary_sampler.name)

            self._has_derived_luminosity = True
            self._derived_luminosity_sampler = auxiliary_sampler

        else:

            assert (
                not auxiliary_sampler.is_secondary
            ), f"{auxiliary_sampler.name} is already set as a secondary sampler!"
            assert (auxiliary_sampler.name not in self._auxiliary_observations
                    ), f"{auxiliary_sampler.name} is already registered!"

            log.info("registering auxilary sampler: %s" %
                     auxiliary_sampler.name)

            self._auxiliary_observations[
                auxiliary_sampler.name] = auxiliary_sampler

    def set_distance_selection(self, selector: SelectionProbabilty) -> None:
        """
        Set the selection type for the distance
        """

        assert isinstance(selector, SelectionProbabilty)

        self._distance_selector = selector

        self._distance_selector_set = True

    def set_flux_selection(self, selector: SelectionProbabilty) -> None:
        """
        Set the selection type for the distance
        """
        assert isinstance(selector, SelectionProbabilty)

        self._flux_selector = selector

        self._flux_selector_set = True

    def add_spatial_selector(self,
                             spatial_selector: SelectionProbabilty) -> None:
        """
        Add a spatial selector into the mix
        """

        assert isinstance(spatial_selector, SelectionProbabilty)

        self._spatial_selector: SelectionProbabilty = spatial_selector

    def _prob_det(self, x: np.ndarray, boundary: float,
                  strength: float) -> np.ndarray:
        """
        Soft detection threshold

        :param x: values to test
        :param boundary: mean value of the boundary
        :param strength: the strength of the threshold
        """

        return sf.expit(strength * (x - boundary))

    @property
    def name(self) -> str:
        return self._name

    def draw_log10_fobs(self, f, f_sigma, size=1) -> np.ndarray:
        """
        draw the log10 of the the fluxes
        """

        log10_f = np.log10(f)

        # sample from the log distribution to keep positive fluxes
        log10_fobs = log10_f + np.random.normal(
            loc=0, scale=f_sigma, size=size)

        return log10_fobs

    def draw_log_fobs(self, f, f_sigma, size=1) -> np.ndarray:
        """
        draw the log10 of the the fluxes
        """

        log_f = np.log(f)

        # sample from the log distribution to keep positive fluxes
        log_fobs = log_f + np.random.normal(loc=0, scale=f_sigma, size=size)

        return log_fobs

    def draw_survey(
        self,
        boundary: float,
        flux_sigma: float = 1.0,
        strength: float = 10.0,
        hard_cut: bool = False,
        distance_probability: Optional[float] = None,
        no_selection: bool = False,
        log10_flux_draw: bool = True,
    ) -> Population:
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
        truth = dict()  # type: dict

        # store the spatial distribution truths
        truth[
            self._spatial_distribution.name] = self._spatial_distribution.truth

        # set the random seed

        np.random.seed(self._seed)

        # create a callback of the integrand
        dNdr = (lambda r: self._spatial_distribution.dNdV(
            r) * self._spatial_distribution.differential_volume(r) / self.
            _spatial_distribution.time_adjustment(r))

        # integrate the population to determine the true number of
        # objects
        N = integrate.quad(dNdr, 0.0,
                           self._spatial_distribution.r_max)[0]  # type: float

        log.info("The volume integral is %f" % N)

        # this should be poisson distributed
        n = np.random.poisson(N)  # type: np.int64

        self._spatial_distribution.draw_distance(size=n)

        # now draw the sky positions

        self._spatial_distribution.draw_sky_positions(size=n)

        distances = self._spatial_distribution.distances  # type: np.ndarray

        log.info("Expecting %d total objects" % n)

        # first check if the auxilliary samplers
        # compute the luminosities

        #      pbar.update()

        # setup the global selection

        global_selection = UnitySelection()  # type: SelectionProbabilty
        global_selection.draw(n)

        # now we set up the selection that _may_ come
        # from the auxilliary samplers

        auxiliary_selection = UnitySelection()  # type: SelectionProbabilty
        auxiliary_selection.draw(n)

        auxiliary_quantities = {}  # type: dict

        # this means the luminosity is not
        # simulated directy

        if self.luminosity_distribution is None:

            assert self._has_derived_luminosity

        if self._has_derived_luminosity:

            # pbar.set_description(desc='Getting derived luminosities')
            # set the distance to the auxilary sampler
            self._derived_luminosity_sampler.set_spatial_values(
                self._spatial_distribution.spatial_values)

            # sample the true and obs
            # values which are held internally

            self._derived_luminosity_sampler.draw(size=n)

            # check to make sure we sampled!
            assert (self._derived_luminosity_sampler.true_values is not None
                    and len(self._derived_luminosity_sampler.true_values) == n)

            assert (self._derived_luminosity_sampler.obs_values is not None
                    and len(self._derived_luminosity_sampler.obs_values) == n)

            # append these values to a dict
            auxiliary_quantities[self._derived_luminosity_sampler.name] = {
                "true_values": self._derived_luminosity_sampler.true_values,
                "obs_values": self._derived_luminosity_sampler.obs_values,
                "selection": self._derived_luminosity_sampler.selector,
            }

            log.info("Getting luminosity from derived sampler")

            luminosities = (
                self._derived_luminosity_sampler.compute_luminosity()
            )  # type: np.ndarray

            # collect anything that was sampled here

            # store the truth from the derived lum sampler

            truth[self._derived_luminosity_sampler.
                  name] = self._derived_luminosity_sampler.truth

            for k2, v2 in self._derived_luminosity_sampler.secondary_samplers.items(
            ):

                # first we tell the sampler to go and retrieve all of
                # its own secondaries

                properties = v2.get_secondary_properties()  # type: dict

                for k3, v3 in properties.items():

                    # now attach them
                    auxiliary_quantities[k3] = v3

                # store the secondary truths
                # this will _could_ be clobbered later
                # but that is ok

                truth[v2.name] = v2.truth

        else:

            luminosities = self.luminosity_distribution.draw_luminosity(
                size=n)  # type: np.ndarray

            # store the truths from the luminosity distribution
            truth[self.luminosity_distribution.
                  name] = self.luminosity_distribution.truth

        # transform the fluxes
        fluxes = self._spatial_distribution.transform(
            luminosities, distances)  # type: np.ndarray

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
            v.set_spatial_values(self._spatial_distribution.spatial_values)

            # sample the true and obs
            # values which are held internally
            # this will also invoke secondary samplers

            v.draw(size=n)

            # store the auxilliary truths
            truth[v.name] = v.truth

            # check to make sure we sampled!
            assert v.true_values is not None and len(v.true_values) == n
            assert v.obs_values is not None and len(v.obs_values) == n

            # append these values to a dict
            auxiliary_quantities[k] = {
                "true_values": v.true_values,
                "obs_values": v.obs_values,
                "selection": v.selector,
            }  # type: dict

            # collect the secondary values

            for k2, v2 in v.secondary_samplers.items():

                # first we tell the sampler to go and retrieve all of
                # its own secondaries

                properties = v2.get_secondary_properties()  # type: dict

                for k3, v3 in properties.items():

                    # now attach them
                    auxiliary_quantities[k3] = v3

                # store the secondary truths

                truth[v2.name] = v2.truth

        # pbar.update()

        # now draw all the observed fluxes
        # this is homoskedastic for now

        if log10_flux_draw:

            log10_fluxes_obs = self.draw_log10_fobs(fluxes, flux_sigma,
                                                    size=n)  # type: np.ndarray
            flux_obs = np.power(10, log10_fluxes_obs)

        else:

            log10_fluxes_obs = self.draw_log_fobs(fluxes, flux_sigma,
                                                  size=n)  # type: np.ndarray

            flux_obs = np.exp(log10_fluxes_obs)

        assert np.alltrue(np.isfinite(log10_fluxes_obs))

        # now select them

        if not no_selection:

            if not self._flux_selector_set:

                DeprecationWarning(
                    "this interface will change and soon you will be required to set the flux selection manually"
                )

                if not hard_cut:

                    self._flux_selector = SoftFluxSelection(boundary, strength)

                    log.info("Applying soft boundary")

                else:

                    log.info("Applying hard boundary")

                    self._flux_selector = HardFluxSelection(boundary)

            # the hard and soft flux selectors have built in
            # properties to let us know what type of selection
            # was made so we can record it

            if isinstance(self._flux_selector,
                          HardFluxSelection) or isinstance(
                              self._flux_selector, SoftFluxSelection):

                if self._flux_selector.hard_cut:

                    strength = 1.0

                else:

                    strength = self._flux_selector.strength

                boundary = self._flux_selector.boundary
                hard_cut = self._flux_selector.hard_cut

            else:

                # These are just dummies for other types of flux selection

                strength = 1
                boundary = 1e-99
                hard_cut = True

        else:

            # These are just dummies for the no selection case

            strength = 1
            boundary = 1e-99
            hard_cut = True

        # pass the values the plux selector and draw the selection
        self._flux_selector.set_observed_flux(flux_obs)

        self._flux_selector.draw(n)

        #       selection = self._flux_selector.selection

        # now apply the selection from the auxilary samplers

        for k, v in auxiliary_quantities.items():

            auxiliary_selection += v["selection"]

            log.info(
                "Applying selection from %s which selected %d of %d objects" %
                (k, v["selection"].n_selected, v["selection"].n_objects))

            log.info(
                "Before auxiliary selection there were %d objects selected" %
                self._flux_selector.n_selected)

        # now we can add the values onto the global
        # selection
        # not in the future we will depreciate the
        # no selection feature
        if not no_selection:

            global_selection += auxiliary_selection

            global_selection += self._flux_selector

            # now scan the spatial selector

            if (self._spatial_selector is not None) and (not no_selection):

                self._spatial_selector.set_spatial_distribution(
                    self._spatial_distribution)

                self._spatial_selector.draw(n)

                log.info(
                    f"Appling selection from {self._spatial_selector.name} which selected {self._spatial_selector.n_selected} of {self._spatial_selector.n_objects}"
                )

                global_selection += self._spatial_selector

        if global_selection.n_selected == n:

            log.warning("NO HIDDEN OBJECTS")

        if not self._distance_selector_set:

            if (distance_probability is not None) or (distance_probability
                                                      == 1.0):

                self._distance_selector = BernoulliSelection(
                    distance_probability)

        self._distance_selector.draw(size=global_selection.n_selected)

        known_distances = distances[global_selection.selection][
            self._distance_selector.selection]
        known_distance_idx = self._distance_selector.selection_index
        unknown_distance_idx = self._distance_selector.non_selection_index

        log.info("Detected %d distances" % len(known_distances))

        try:

            log.info("Deteced %d objects our to a distance of %.2f" %
                     (global_selection.n_selected, max(known_distances)))

        except:

            log.warning("No Objects detected")

        # just to make sure we do not do anything nutty
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
            flux_obs=flux_obs,
            selection=global_selection.selection,
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
            theta=self._spatial_distribution.theta,
            phi=self._spatial_distribution.phi,
        )

    def display(self) -> None:
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

            self._graph.add_edge(self._derived_luminosity_sampler.name,
                                 "obs_flux")

            if self._derived_luminosity_sampler.uses_distance:

                self._graph.add_edge(
                    self._spatial_distribution.name,
                    self._derived_luminosity_sampler.name,
                )

            for k2, v2 in self._derived_luminosity_sampler.secondary_samplers.items(
            ):

                self._graph.add_node(k2)
                self._graph.add_edge(k2, self._derived_luminosity_sampler.name)

                # pass the graph and the primary

                _ = v2.get_secondary_properties(
                    graph=self._graph,
                    primary=k2,
                    spatial_distribution=self._spatial_distribution,
                )

        else:
            self._graph.add_edge(self._luminosity_distribution.name,
                                 "obs_flux")
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
