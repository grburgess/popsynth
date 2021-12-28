from abc import ABCMeta
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import networkx as nx
import numpy as np
import pandas as pd
import scipy.integrate as integrate

import yaml
from IPython.display import Markdown, Math, display

# from numpy.typing import np.ndarray

from popsynth.auxiliary_sampler import (
    AuxiliarySampler,
    DerivedLumAuxSampler,
    SecondaryContainer,
    SecondaryStorage,
)
from popsynth.distribution import LuminosityDistribution, SpatialDistribution
from popsynth.distributions.cosmological_distribution import CosmologicalDistribution
from popsynth.population import Population
from popsynth.selection_probability import SelectionProbability, UnitySelection
from popsynth.utils.logging import setup_logger
from popsynth.utils.registry import (
    auxiliary_parameter_registry,
    distribution_registry,
    selection_registry,
)

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

        :param spatial_distribution: The spatial distribution to sample locations from
        :type spatial_distribution: :class:`SpatialDistribution`
        :param luminosity_distribution: The optional luminosity distribution
        :type luminosity_distribution: :class:`LuminosityDistribution`
        :param seed: Random seed
        :type seed: int
        """

        self._n_model = 500  # type: int
        self._seed = int(seed)  # type: int

        # this is a container for things passed to
        # stan
        self._model_spaces = {}  # type: dict

        self._auxiliary_observations: Dict[str, AuxiliarySampler] = {}

        self._graph = nx.DiGraph()  # type: nx.Digraph

        if not isinstance(spatial_distribution, SpatialDistribution):

            log.error("the spatial_distribution is the wrong type")

            raise RuntimeError()

        self._name = f"{spatial_distribution.name}"  # type: str

        if luminosity_distribution is not None:

            if not isinstance(luminosity_distribution, LuminosityDistribution):

                log.error("the luminosity_distribution is the wrong type")

                raise RuntimeError()

            self._name = f"{self._name}_{luminosity_distribution.name}"

        self._spatial_distribution = spatial_distribution  # type: SpatialDistribution
        self._luminosity_distribution = (
            luminosity_distribution
        )  # type: Union[LuminosityDistribution, None]

        self._has_derived_luminosity = False  # type: bool
        self._derived_luminosity_sampler = (
            None)  # type: Union[DerivedLumAuxSampler, None]

        # set the selections be fully seen unless it is set by the user
        self._distance_selector: SelectionProbability = UnitySelection(
            name="unity distance selector")
        self._flux_selector: SelectionProbability = UnitySelection(
            name="unity flux selector")

        # check to see if the selectors are set
        self._distance_selector_set: bool = False
        self._flux_selector_set: bool = False
        self._spatial_selector: Optional[SelectionProbability] = None

        self._params = {}  # type: dict

        # keep a list of parameters here for checking

        for k, v in self._spatial_distribution.params.items():

            self._params[k] = v

        if self._luminosity_distribution is not None:
            for k, v in self._luminosity_distribution.params.items():

                self._params[k] = v

        self._graph.add_node(self._spatial_distribution.name)

        # add the sky sampler

    def clean(self, reset: bool = False):
        """
        Clean the auxiliary samplers, selections, etc
        from the population synth

        :param reset: If `True`, reset any attached distributions and samplers
        :type reset: bool
        """

        if reset:

            for k, v in self._auxiliary_observations.items():

                v.reset()

        log.warning("removing all registered Auxiliary Samplers")

        self._auxiliary_observations = {}

        if reset:

            if self._derived_luminosity_sampler is not None:

                self._derived_luminosity_sampler.reset()

        self._derived_luminosity_sampler = None

        self._has_derived_luminosity = False

        self._distance_selector_set = False

        self._flux_selector_set = False

        log.warning("removing flux selector")

        if reset:

            self._flux_selector.reset()

        self._flux_selector = UnitySelection(name="unity flux selector")

        log.warning("removing distance selector")

        if reset:

            self._distance_selector.reset()

        self._distance_selector = UnitySelection(
            name="unity distance selector")

        log.warning("removing spatial selector")

        if reset:

            if self._spatial_selector is not None:

                self._spatial_selector.reset()

        self._spatial_selector = None

    def write_to(self, file_name: str) -> None:
        """
        Write the population synth to a YAML file.

        :param file_name: the file name of the output YAML
        :type file_name: str
        """
        with open(file_name, "w") as f:

            yaml.dump(
                stream=f,
                data=self.to_dict(),
                default_flow_style=False,
                Dumper=yaml.SafeDumper,
            )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the population synth to a dictionary

        :returns: Popsynth dict
        :rtype: Dict[str, Any]
        """

        output: Dict[str, Any] = {}

        output["seed"] = self._seed

        # store the spatial distribution

        spatial_distribution = {}

        spatial_distribution[
            self._spatial_distribution.
            _distribution_name] = self._spatial_distribution.truth

        # store is_rate if cosmological distribution

        if isinstance(self._spatial_distribution, CosmologicalDistribution):

            spatial_distribution[
                "is_rate"] = self._spatial_distribution._is_rate

        output["spatial distribution"] = spatial_distribution

        # if there is a luminosity distribution
        # then get it and store it

        if self._luminosity_distribution is not None:

            luminosity_distribution = {}

            luminosity_distribution[
                self._luminosity_distribution.
                _distribution_name] = self._luminosity_distribution.truth

            output["luminosity distribution"] = luminosity_distribution

        if self._flux_selector_set:

            flux_selection = {}

            flux_selection[self._flux_selector.
                           _selection_name] = self._flux_selector.parameters

            output["flux selection"] = flux_selection

        if self._distance_selector_set:

            distance_selection = {}

            distance_selection[
                self._distance_selector.
                _selection_name] = self._distance_selector.parameters

            output["distance selection"] = distance_selection

        if self._spatial_selector is not None:

            spatial_selection = {}

            spatial_selection[
                self._spatial_selector.
                _selection_name] = self._spatial_selector.parameters

            output["spatial selection"] = spatial_selection

        aux_samplers = {}

        for k, v in self._auxiliary_observations.items():

            tmp = {}

            tmp["type"] = v._auxiliary_sampler_name
            tmp["observed"] = v.observed

            for k2, v2 in v.truth.items():

                tmp[k2] = v2

            tmp["secondary"] = list(v.secondary_samplers.keys())

            selection = {}
            selection[v.selector._selection_name] = v.selector.parameters

            tmp["selection"] = selection

            aux_samplers[k] = tmp

            if v.has_secondary:

                aux_samplers = v.get_secondary_objects(aux_samplers)

        output["auxiliary samplers"] = aux_samplers

        return output

    @classmethod
    def from_dict(cls, input: Dict[str, Any]) -> "PopulationSynth":
        """
        Build a PopulationSynth object from a dictionary

        :param input: the dictionary from which to build
        :type input: Dict[str, Any]
        :returns: Popsynth object
        :rtype: :class:`PopulationSynth`
        """

        if "luminosity distribution" in input:

            # try:

            # load the name and parameters

            tmp = input["luminosity distribution"]

            ld_name = list(tmp.keys())[0]

            # create the instance

            luminosity_distribution: LuminosityDistribution = distribution_registry[
                ld_name]

            # now set the values of the parameters

            log.debug(f"setting parameters for {ld_name}")

            for k, v in tmp[ld_name].items():

                log.debug(f"trying to set {k} to {v}")

                for x in luminosity_distribution.__class__.mro():

                    if k in x.__dict__:

                        setattr(luminosity_distribution, k, float(v))

                        break

                log.debug(f"{luminosity_distribution.params}")

        else:

            luminosity_distribution = None

        # try

        tmp = input["spatial distribution"]

        sd_name = list(tmp.keys())[0]

        spatial_distribution: SpatialDistribution = distribution_registry[
            sd_name]

        for k, v in tmp[sd_name].items():

            log.debug(f"trying to set {k} to {v}")

            for x in spatial_distribution.__class__.mro():

                if k in x.__dict__:

                    setattr(spatial_distribution, k, float(v))

                    break

        if isinstance(spatial_distribution, CosmologicalDistribution):

            spatial_distribution._is_rate = tmp["is_rate"]

        seed: int = input["seed"]

        # create the poopulation synth

        pop_synth: PopulationSynth = cls(
            spatial_distribution,
            luminosity_distribution=luminosity_distribution,
            seed=seed,
        )

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

                fs = selection_registry.get(fs_name)

                for k, v in params.items():

                    log.debug(f"trying to set {k} to {v}")

                    for x in fs.__class__.mro():

                        if k in x.__dict__:

                            setattr(fs, k, float(v))

                            break

                # oh yes we are doing that
                fs._use_flux = True

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

                ds = selection_registry.get(ds_name)

                for k, v in params.items():

                    log.debug(f"trying to set {k} to {v}")

                    for x in ds.__class__.mro():

                        if k in x.__dict__:

                            setattr(ds, k, float(v))

                            break

                ds._use_distance = True

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

                ss = selection_registry.get(ss_name)

                # make sure they are all floats

                for k, v in params.items():

                    log.debug(f"trying to set {k} to {v}")

                    for x in ss.__class__.mro():

                        if k in x.__dict__:

                            setattr(ss, k, float(v))

                            break

                pop_synth.add_spatial_selector(ss)

        # Now collect the auxiliary samplers

        # we will gather them so that we can sort out dependencies
        aux_samplers: Dict[str, AuxiliarySampler] = OrderedDict()
        secondary_samplers: [str, str] = OrderedDict()

        if "auxiliary samplers" in input:

            for obj_name, v in input["auxiliary samplers"].items():

                # first we extract the required info

                sampler_name = v.pop("type")
                is_observed = v.pop("observed")

                # now we extract the selection
                # and secondary if they are there

                log.debug(
                    f"starting to scan {obj_name} of type {sampler_name}")

                if "selection" in v:
                    selection = v.pop("selection")

                else:

                    selection = None

                if "secondary" in v:

                    log.debug(f"auxiliary sampler {obj_name} has secondaries")

                    secondary = v.pop("secondary")

                    if isinstance(secondary, dict):

                        secondary = list(np.atleast_1d(list(secondary.keys())))

                    else:
                        secondary = list(np.atleast_1d(secondary))

                    log.debug(f"secondaries are {secondary}")

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
                        **init_variables,
                    )

                except (TypeError):

                    # try without name

                    try:

                        tmp: AuxiliarySampler = auxiliary_parameter_registry.get(
                            sampler_name,
                            # name=obj_name,
                            observed=is_observed,
                            **init_variables,
                        )

                    except (TypeError):

                        try:

                            tmp: AuxiliarySampler = auxiliary_parameter_registry.get(
                                sampler_name,
                                name=obj_name,
                                # observed=is_observed,
                                **init_variables,
                            )

                        except (TypeError):

                            tmp: AuxiliarySampler = auxiliary_parameter_registry.get(
                                sampler_name, **init_variables)

                            # we will do a dirty trick
                            tmp._name = obj_name
                            tmp._is_observed = is_observed

                log.debug(f"setting parameters for {sampler_name}: {obj_name}")

                # now set the parameters

                for k, v in params.items():

                    log.debug(f"trying to set {k} to {v}")

                    for x in tmp.__class__.mro():

                        if k in x.__dict__:

                            setattr(tmp, k, float(v))

                            break

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

                    selector = selection_registry.get(sel_name)

                    for k, v in params.items():

                        if k in selector.__class__.__dict__:

                            setattr(selector, k, float(v))

                    selector._use_obs_value = True

                    tmp.set_selection_probability(selector)

                # now we store this sampler

                log.debug(f"{obj_name} built")

                aux_samplers[obj_name] = tmp

                # if there is a secondary sampler,
                # we need to make a mapping

                if secondary is not None:

                    log.debug(
                        f"{obj_name} is adding {secondary} as secondaries")

                    secondary_samplers[obj_name] = secondary

        # Now we have collected all of the auxiliary samplers
        # we need to assign those which are secondary

        log.debug(f"have {list(aux_samplers.keys())} as aux samplers")

        for primary, secondaries in secondary_samplers.items():

            for secondary in secondaries:

                if secondary is None:

                    break

                # assign it
                aux_samplers[primary].set_secondary_sampler(
                    aux_samplers[secondary])

        # now we need to pop all the secondaries from the main
        # list so that we do not double add

        for secondaries in list(secondary_samplers.values()):

            for secondary in secondaries:

                if secondary in aux_samplers:

                    aux_samplers.pop(secondary)

        # now we add the observed quantites on

        for k, v in aux_samplers.items():

            pop_synth.add_observed_quantity(v)

        return pop_synth

    @classmethod
    def from_file(cls, file_name: str) -> "PopulationSynth":
        """
        read the population in from a yaml file

        :param file_name: the file name of the population synth
        """
        with open(file_name) as f:

            input: Dict[str, Any] = yaml.load(f, Loader=yaml.SafeLoader)

        return cls.from_dict(input)

    @property
    def spatial_distribution(self) -> SpatialDistribution:
        return self._spatial_distribution

    @property
    def luminosity_distribution(self) -> Union[LuminosityDistribution, None]:
        return self._luminosity_distribution

    def add_model_space(self, name, start, stop, log=True):
        """
        Add a model space for stan generated quantities

        :param name: Name that Stan will use
        :param start: Start of the grid
        :param stop: Stop of the grid
        :param log: Use log10 or not
        """
        if log:
            space = np.logspace(np.log10(start), np.log10(stop), self._n_model)

        else:

            space = np.linspace(start, stop, self._n_model)

        self._model_spaces[name] = space

    def add_auxiliary_sampler(self,
                              auxiliary_sampler: Union[DerivedLumAuxSampler,
                                                       AuxiliarySampler]):
        """
        Add an auxiliary sampler or derived luminosity sampler to the population
        synth.

        :param auxiliary_sampler: The auxiliary_sampler
        :type auxiliary_sampler: Union[DerivedLumAuxSampler, AuxiliarySampler]
        """

        self.add_observed_quantity(auxiliary_sampler)

    def add_observed_quantity(self,
                              auxiliary_sampler: Union[DerivedLumAuxSampler,
                                                       AuxiliarySampler]):
        """
        Add an auxiliary sampler or derived luminosity sampler to the population
        synth

        :param auxiliary_sampler: The auxiliary_sampler
        :type auxiliary_sampler: Union[DerivedLumAuxSampler, AuxiliarySampler]
        """

        if isinstance(auxiliary_sampler, DerivedLumAuxSampler):

            log.info(
                f"registering derived luminosity sampler: {auxiliary_sampler.name}"
            )

            self._has_derived_luminosity = True
            self._derived_luminosity_sampler = auxiliary_sampler

        elif isinstance(auxiliary_sampler, AuxiliarySampler):

            if auxiliary_sampler.is_secondary:
                log.error(
                    f"{auxiliary_sampler.name} is already set as a secondary sampler!"
                )
                log.error(
                    f"and registered to {','.join(auxiliary_sampler.parents)}")

                raise RuntimeError()

            if auxiliary_sampler.name in self._auxiliary_observations:

                log.error(f"{auxiliary_sampler.name} is already registered!")

                raise RuntimeError()

            log.info("registering auxilary sampler: %s" %
                     auxiliary_sampler.name)

            self._auxiliary_observations[
                auxiliary_sampler.name] = auxiliary_sampler

        else:

            log.error("This not an auxiliary sampler")

            raise RuntimeError()

    def set_distance_selection(self, selector: SelectionProbability) -> None:
        """
        Set the selection type for the distance.

        :param selector: The selector
        :type selector: :class:`SelectionProbability`
        """

        if not isinstance(selector, SelectionProbability):

            log.error(f"{selector} is not a Selection probability")

            raise RuntimeError()

        self._distance_selector = selector

        self._distance_selector_set = True

    def set_flux_selection(self, selector: SelectionProbability) -> None:
        """
        Set the selection type for the flux

        :param selector: The selector
        :type selector: :class:`SelectionProbability`
        """
        if not isinstance(selector, SelectionProbability):

            log.error(f"{selector} is not a Selection probability")

            raise RuntimeError()

        self._flux_selector = selector

        self._flux_selector_set = True

    def add_spatial_selector(self,
                             spatial_selector: SelectionProbability) -> None:
        """
        Add a spatial selector into the mix

        :param spatial_selector: The spatial selector
        :type spatial_selector: :class:`SelectionProbability`
        """

        if not isinstance(spatial_selector, SelectionProbability):

            log.error(f"{spatial_selector} is not a Selection probability")

            raise RuntimeError()

        self._spatial_selector = spatial_selector

    @property
    def name(self) -> str:
        return self._name

    def draw_log10_fobs(self, f, f_sigma, size=1) -> np.ndarray:
        """
        Draw the log10 of the the fluxes.
        """

        log10_f = np.log10(f)

        # sample from the log distribution to keep positive fluxes
        log10_fobs = log10_f + np.random.normal(
            loc=0, scale=f_sigma, size=size)

        return log10_fobs

    def draw_log_fobs(self, f, f_sigma, size=1) -> np.ndarray:
        """
        Draw the log10 of the the fluxes.
        """

        log_f = np.log(f)

        # sample from the log distribution to keep positive fluxes
        log_fobs = log_f + np.random.normal(loc=0, scale=f_sigma, size=size)

        return log_fobs

    def draw_survey(
        self,
        flux_sigma: Optional[float] = None,
        log10_flux_draw: bool = True,
    ) -> Population:
        """
        Draw the total survey and return a :class:`Population` object.

        This will sample all attached distributions and apply selection functions.

        If a value of flux_sigma is given, the log10 observed fluxes are sampled with
        measurement error.

        :param flux_sigma: The homoskedastic sigma for the flux in log10 space
        :type flux_sigma: Optional[float]
        :param log10_flux_draw: if `True`, fluxes are drawn in log space
        :type log10_flux_draw: bool
        :return: a Population object
        :rtype: :class:`Population`
        """

        # this stores all the "true" population values from all the samplers
        truth = dict()  # type: dict

        # store the spatial distributVion truths
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

        # setup the global selection

        global_selection: SelectionProbability = UnitySelection(name="global")
        global_selection.select(n)

        # now we set up the selection that _may_ come
        # from the auxilliary samplers

        auxiliary_selection: SelectionProbability = UnitySelection(
            name="total auxiliary selection")
        auxiliary_selection.select(n)

        auxiliary_quantities: SecondaryStorage = SecondaryStorage()

        # this means the luminosity is not
        # simulated directy

        if self.luminosity_distribution is None:

            if not self._has_derived_luminosity:

                log.error("No luminosity distribution was specified")
                log.error(
                    "and no derived luminosity auxiliary sampler was added")
                raise RuntimeError()

        if self._has_derived_luminosity:

            log.debug("using a derived luminosity sampler")
            # pbar.set_description(desc='Getting derived luminosities')
            # set the distance to the auxilary sampler
            self._derived_luminosity_sampler.set_spatial_values(
                self._spatial_distribution.spatial_values)

            # sample the true and obs
            # values which are held internally

            self._derived_luminosity_sampler.draw(size=n)

            log.debug("derived luminosity sampled")

            # check to make sure we sampled!
            assert (self._derived_luminosity_sampler.true_values is not None
                    and len(self._derived_luminosity_sampler.true_values) == n)

            assert (self._derived_luminosity_sampler.obs_values is not None
                    and len(self._derived_luminosity_sampler.obs_values) == n)

            # append these values to a dict

            auxiliary_quantities.add_secondary(
                SecondaryContainer(
                    self._derived_luminosity_sampler.name,
                    self._derived_luminosity_sampler.true_values,
                    self._derived_luminosity_sampler.obs_values,
                    self._derived_luminosity_sampler.selector,
                ))

            log.info("Getting luminosity from derived sampler")

            luminosities = (
                self._derived_luminosity_sampler.compute_luminosity()
            )  # type: np.ndarray

            # collect anything that was sampled here

            # store the truth from the derived lum sampler

            truth[self._derived_luminosity_sampler.
                  name] = self._derived_luminosity_sampler.truth

            log.debug("sampling ")

            for k2, v2 in self._derived_luminosity_sampler.secondary_samplers.items(
            ):

                # first we tell the sampler to go and retrieve all of
                # its own secondaries

                auxiliary_quantities += v2.get_secondary_properties()

                #                properties = v2.get_secondary_properties()  # type: dict

                # for k3, v3 in properties.items():

                #     # now attach them
                #     auxiliary_quantities[k3] = v3

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

        for k, v in self._auxiliary_observations.items():

            assert (
                not v.is_secondary
            ), "This is a secondary sampler. You cannot sample it in the main sampler"

            # set the luminosities and distances to
            # auxilary sampler just in case
            # they are needed

            v.set_luminosity(luminosities)
            v.set_spatial_values(self._spatial_distribution.spatial_values)

            # also set luminosities and distances to secondaries
            # as needed

            for k2, v2 in v.secondary_samplers.items():

                if v2.uses_luminosity:

                    v2.set_luminosity(luminosities)

                if v2.uses_distance:

                    v2.set_spatial_values(
                        self._spatial_distribution.spatial_values)

            # sample the true and obs
            # values which are held internally
            # this will also invoke secondary samplers

            v.draw(size=n)

            # store the auxilliary truths
            truth[v.name] = v.truth

            # check to make sure we sampled!
            assert v.true_values is not None and len(v.true_values) == n
            assert v.obs_values is not None and len(v.obs_values) == n

            auxiliary_quantities += v.get_secondary_properties()

            # # append these values to a dict
            # auxiliary_quantities[k] = {
            #     "true_values": v.true_values,
            #     "obs_values": v.obs_values,
            #     "selection": v.selector,
            # }  # type: dict

            # collect the secondary values

            for k2, v2 in v.secondary_samplers.items():

                # first we tell the sampler to go and retrieve all of
                # its own secondaries

                # properties = v2.get_secondary_properties()  # type: dict

                # for k3, v3 in properties.items():

                #     # now attach them
                #     auxiliary_quantities[k3] = v3

                # store the secondary truths

                truth[v2.name] = v2.truth

        # pbar.update()

        # now draw all the observed fluxes
        # this is homoskedastic for now

        if not isinstance(self._flux_selector, UnitySelection):

            if flux_sigma is not None:

                log.debug("assuming that fluxes will be jittered")

                if log10_flux_draw:

                    log.debug("making a log10 flux draw")

                    log10_fluxes_obs = self.draw_log10_fobs(
                        fluxes, flux_sigma, size=n)  # type: np.ndarray
                    flux_obs = np.power(10, log10_fluxes_obs)

                else:

                    log.debug("making a logflux draw")

                    log10_fluxes_obs = self.draw_log_fobs(
                        fluxes, flux_sigma, size=n)  # type: np.ndarray

                    flux_obs = np.exp(log10_fluxes_obs)

                assert np.alltrue(np.isfinite(log10_fluxes_obs))

            else:

                log.debug("observed fluxes are latent fluxes")

                flux_obs = fluxes
                log10_fluxes_obs = np.log10(fluxes)
                flux_sigma = -1  # this is a dummy

        else:

            log.debug("observed fluxes are latent fluxes")

            flux_obs = fluxes
            log10_fluxes_obs = np.log10(fluxes)
            flux_sigma = -1  # this is a dummy

        log.info("applying selection to fluxes")

        # pass the values the plux selector and draw the selection
        self._flux_selector.set_observed_flux(flux_obs)

        self._flux_selector.select(n)

        #       selection = self._flux_selector.selection

        # now apply the selection from the auxilary samplers

        for k, v in auxiliary_quantities.items():

            # unity selections don't add anything

            if isinstance(v["selection"], UnitySelection):

                log.debug(f"skipping {k} selection because it is unity")

                continue

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

        global_selection += auxiliary_selection

        global_selection += self._flux_selector

        # now scan the spatial selector

        if self._spatial_selector is not None:

            self._spatial_selector.set_spatial_distribution(
                self._spatial_distribution)

            self._spatial_selector.select(n)

            log.info(
                f"Appling selection from {self._spatial_selector.name} which selected {self._spatial_selector.n_selected} of {self._spatial_selector.n_objects}"
            )

            global_selection += self._spatial_selector

        if global_selection.n_selected == n:

            log.warning("NO HIDDEN OBJECTS")

        self._distance_selector.select(size=global_selection.n_selected)

        known_distances = distances[global_selection.selection][
            self._distance_selector.selection]
        known_distance_idx = self._distance_selector.selection_index
        unknown_distance_idx = self._distance_selector.non_selection_index

        log.info(f"Detected {len(known_distances)} distances")

        try:

            log.info(
                f"Detected {global_selection.n_selected} objects out to a distance of {max(known_distances):.2f}"
            )

        except:

            log.warning("No Objects detected")

        # just to make sure we do not do anything nutty
        lf_params = None
        lf_form = None
        if self._luminosity_distribution is not None:

            lf_params = self._luminosity_distribution.params
            lf_form = self._luminosity_distribution.form

        # if distance_probability is None:
        #     distance_probability = 1.0

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
            seed=self._seed,
            name=self._name,
            spatial_form=self._spatial_distribution.form,
            lf_form=lf_form,
            auxiliary_quantities=auxiliary_quantities,
            truth=truth,
            graph=self.graph,
            theta=self._spatial_distribution.theta,
            phi=self._spatial_distribution.phi,
            pop_synth=self.to_dict(),
        )

    def display(self) -> None:
        """
        Display the simulation parameters.
        """

        if self._has_derived_luminosity:

            display(Markdown("## Luminosity Function"))

            self._derived_luminosity_sampler.display()

        elif self._luminosity_distribution is not None:

            display(Markdown("## Luminosity Function"))

            self._luminosity_distribution.display()

        display(Markdown("## Spatial Function"))

        self._spatial_distribution.display()

        names = []

        if self._has_derived_luminosity:

            for k, v in self._derived_luminosity_sampler.secondary_samplers.items(
            ):

                names.append(k)

                display(Markdown(f"## {k}"))

                v.display()

        for k, v in self._auxiliary_observations.items():

            if k not in names:

                display(Markdown(f"## {k}"))

                v.display()

    def __repr__(self) -> str:

        if self._has_derived_luminosity:

            out = "Luminosity Function\n"

            out += self._derived_luminosity_sampler.__repr__()

        elif self._luminosity_distribution is not None:

            out = "Luminosity Function\n"

            out += self._luminosity_distribution.__repr__()

        out += "Spatial Function\n"

        out += self._spatial_distribution.__repr__()

        names = []

        if self._has_derived_luminosity:

            for k, v in self._derived_luminosity_sampler.secondary_samplers.items(
            ):

                names.append(k)

                out += v.__repr__()

        for k, v in self._auxiliary_observations.items():

            if k not in names:

                out += v.__repr__()

        return out

    # def generate_stan_code(self, stan_gen, **kwargs):

    #     pass

    @property
    def graph(self):

        self._build_graph()

        return self._graph

    def _build_graph(self):
        """
        Builds the graph for all the samplers.
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
