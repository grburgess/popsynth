from typing import Any, Dict, Optional

import h5py
import ipyvolume as ipv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pythreejs
from betagen import betagen
from IPython.display import Markdown, Math, display
from numpy.typing import ArrayLike

from popsynth.auxiliary_sampler import SecondaryStorage
from popsynth.utils.array_to_cmap import array_to_cmap
from popsynth.utils.hdf5_utils import (
    clean_graph_dict,
    fill_graph_dict,
    recursively_load_dict_contents_from_group,
    recursively_save_dict_contents_to_group,
)
from popsynth.utils.logging import setup_logger
from popsynth.utils.spherical_geometry import xyz

log = setup_logger(__name__)

wine = "#8F2727"
dark, dark_highlight, mid, mid_highlight, light, light_highlight = betagen(
    wine)


class Population(object):

    def __init__(
        self,
        luminosities: ArrayLike,
        distances: ArrayLike,
        known_distances: ArrayLike,
        known_distance_idx: ArrayLike,
        unknown_distance_idx: ArrayLike,
        fluxes: ArrayLike,
        flux_obs: ArrayLike,
        selection: ArrayLike,
        flux_sigma: float,
        r_max: float,
        n_model: int,
        lf_params: Dict[str, Any],
        spatial_params: Optional[Dict[str, Any]] = None,
        model_spaces: Optional[ArrayLike] = None,
        seed: int = 1234,
        name: Optional[str] = None,
        spatial_form: Optional[Dict[str, Any]] = None,
        lf_form: Optional[Dict[str, Any]] = None,
        auxiliary_quantities: Optional[SecondaryStorage] = None,
        truth: Dict[str, float] = {},
        graph: Optional[Dict[str, Any]] = None,
        theta=None,
        phi=None,
        pop_synth: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        A population container for all the properties of the population.

        :param luminosities: The luminosities
        :type luminosities: ArrayLike
        :param distances: The distances
        :type distances: ArrayLike
        :param known_distances: The known distances
        :type known_distances: ArrayLike
        :param known_distance_idx: The index of the known distances
        :type known_distance_idx: ArrayLike
        :param unknown_distance_idx: The index of the unknown distances
        :type unknown_distance_idx: ArrayLike
        :param fluxes: The latent fluxes
        :type fluxes: ArrayLike
        :param flux_obs: The observed fluxes
        :type flux_obs: ArrayLike
        :param selection: The selection vector
        :type selection: ArrayLike
        :param flux_sigma: The uncertainty on the observed flux
        :type flux_sigma: float
        :param r_max: The maximum distance of the survey
        :type r_max: float
        :param n_model: Number of models
        :type n_model: int
        :param lf_params: Luminosity function parameters
        :type lf_params: Dict[str, Any]
        :param spatial_params: Spatial distribution parameters
        :type spatial_params: Optional[Dict[str, Any]]
        :param model_spaces: Model spaces
        :type model_spaces: ArrayLike
        :param seed: Random seed
        :type seed: int
        :param name: Name of the population
        :type name: str
        :param spatial_form: Form of the spatial distribution
        :type spatial_form: Optional[Dict[str, Any]]
        :param lf_form: Form of the luminosity function
        :type lf_form: Optional[Dict[str, Any]]
        :param auxiliary_quantities: Dict of auxiliary quantities
        :type auxiliary_quantities: Optional[Dict[str, Any]]
        :param truth: Dict of true values
        :type truth: Dict[str, float]
        :param graph: Graph of generative model
        :type graph: Optional[Dict[str, Any]]
        :param theta: Theta
        :type theta:
        :param phi: Phi
        :type phi:
        :param pop_synth: Population synth
        :type pop_synth: Optional[Dict[str, Any]]
        """
        self._luminosities = luminosities  # type: ArrayLike

        self._distances = distances  # type: ArrayLike
        self._known_distances = known_distances  # type: ArrayLike
        self._known_distance_idx = known_distance_idx  # type: ArrayLike
        self._unknown_distance_idx = unknown_distance_idx  # type: ArrayLike

        self._theta = theta  # type: ArrayLike
        self._phi = phi  # type: ArrayLike

        assert len(known_distances) + len(unknown_distance_idx) == sum(
            selection), "the distances are not the correct size"

        # latent fluxes
        self._fluxes = fluxes  # type: ArrayLike

        # observed fluxes
        self._flux_obs = flux_obs  # type: ArrayLike
        self._selection = selection  # type: ArrayLike
        self._flux_sigma = flux_sigma  # type: float

        self._r_max = r_max  # type: float
        self._seed = seed  # type: int
        self._n_model = n_model  # type: int
        self._name = name  # type: str
        self._spatial_form = spatial_form
        self._lf_form = lf_form

        self._flux_selected = flux_obs[selection]  # type: ArrayLike
        self._distance_selected = distances[selection]  # type: ArrayLike
        self._luminosity_selected = luminosities[selection]  # type: ArrayLike

        self._flux_hidden = flux_obs[~selection]  # type: ArrayLike
        self._distance_hidden = distances[~selection]  # type: ArrayLike
        self._luminosity_hidden = luminosities[~selection]  # type: ArrayLike

        self._lf_params = lf_params
        self._spatial_params = spatial_params

        self._model_spaces = model_spaces

        self._truth = truth

        self._graph = graph

        self._n_objects = len(selection)  # type: int
        self._n_detections = sum(self._selection)  # type: int
        self._n_non_detections = self._n_objects - self._n_detections  # type: int

        self._pop_synth: Optional[Dict[str, Any]] = pop_synth

        if self._n_detections == 0:

            self._no_detection = True

            log.warning("THERE ARE NO DETECTED OBJECTS IN THE POPULATION")

        else:

            self._no_detection = False

        if auxiliary_quantities is not None:

            for k, v in auxiliary_quantities.items():

                setattr(self, k, v["true_values"])
                setattr(self, "%s_obs" % k, v["obs_values"])
                setattr(self, "%s_selected" % k, v["obs_values"][selection])

        self._auxiliary_quantities = auxiliary_quantities

        if model_spaces is not None:

            for k, v in model_spaces.items():

                assert len(v) == n_model

    @property
    def graph(self):
        """
        The networkx graph of the population

        :returns: 

        """

        return self._graph

    @property
    def pop_synth(self) -> Dict[str, Any]:
        """
        Dictionary population synth used to create this population
        """
        return self._pop_synth

    @property
    def truth(self):
        """
        The simulated truth parameters
        """

        return self._truth

    @property
    def flux_sigma(self) -> float:
        """
        The simulated flux sigma
        """

        return self._flux_sigma

    @property
    def theta(self) -> np.ndarray:
        """
        The polar angle of the objects
        """
        return self._theta

    @property
    def phi(self) -> np.ndarray:
        """
        The phi angle of the objects
        """
        return self._phi

    @property
    def dec(self) -> np.ndarray:
        """
        The declination of the objects
        """
        return 90 - np.rad2deg(self._theta)

    @property
    def ra(self) -> np.ndarray:
        """
        The right ascension of the objects
        """
        return np.rad2deg(self._phi)

    @property
    def luminosities_latent(self) -> np.ndarray:
        """
        The true luminosities of the objects. These are always latent
        as one cannot directly observe them.
        """
        return self._luminosities

    @property
    def distances(self) -> np.ndarray:
        """
        The distances to the objects
        """
        return self._distances

    @property
    def known_distances(self) -> np.ndarray:
        """
        The observed distances
        """

        return self._known_distances

    @property
    def selection(self) -> np.ndarray:
        """
        The selection vector
        """
        return self._selection

    @property
    def fluxes_latent(self) -> np.ndarray:
        """
        The latent fluxes of the objects
        """
        return self._fluxes

    @property
    def fluxes_observed(self) -> np.ndarray:
        """
        All of the observed fluxes, i.e.,
        scattered with error
        """
        return self._flux_obs

    @property
    def selected_fluxes_observed(self) -> np.ndarray:
        """
        The selected obs fluxes
        """

        return self._flux_selected

    @property
    def selected_fluxes_latent(self) -> np.ndarray:
        """
        The selected latent fluxes
        """

        return self._fluxes[self._selection]

    @property
    def selected_distances(self) -> np.ndarray:
        """
        The selected distances. Note, this is different than
        the KNOWN distances.
        """
        return self._distance_selected

    @property
    def hidden_fluxes_observed(self) -> np.ndarray:
        """
        The observed fluxes that are hidden by the selection
        """

        return self._flux_hidden

    @property
    def hidden_distances(self) -> np.ndarray:
        """
        The distances that are hidden by the selection
        """

        return self._distance_hidden

    @property
    def hidden_fluxes_latent(self) -> np.ndarray:
        """
        The latent fluxes that are hidden by the selection
        """

        return self._fluxes[~self._selection]

    @property
    def hard_cut(self) -> bool:
        return self._hard_cut

    @property
    def distance_probability(self) -> float:
        return self._distance_probability

    @property
    def luminosity_parameters(self):
        return self._lf_params

    @property
    def n_objects(self) -> int:
        """
        The number of objects in the population
        """
        return self._n_objects

    @property
    def n_detections(self) -> int:
        """
        The number of DETECTED objects in the population
        """

        return self._n_detections

    @property
    def n_non_detections(self) -> int:
        """
        The number of NON-DETECTED objects in the population
        """

        return self._n_non_detections

    @property
    def has_detections(self) -> bool:
        """
        If the population has detections
        """

        return not self._no_detection

    @property
    def spatial_parameters(self):
        """
        spatial parameters

        :returns: 

        """

        return self._spatial_params

    def __repr__(self):

        out = ""
        out += f"total number of objects: {self.n_detections + self.n_non_detections}"
        out += f"\n number of detections: {self.n_detections}"
        out += f"\n number of non-detections: {self.n_non_detections}"

        return out

    def to_stan_data(self) -> dict:
        """
        Create Stan input
        """

        # create a dict for Stan
        output = dict(
            N=self._n_detections,
            Nz=len(self._known_distances),
            Nnz=len(self._unknown_distance_idx),
            z_obs=self._distance_selected,
            known_z_obs=self._known_distances,
            z_idx=self._known_distance_idx + 1,  # stan indexing
            z_nidx=self._unknown_distance_idx + 1,  # stan indexing
            r_obs=self._distance_selected,
            known_r_obs=self._known_distances,
            r_idx=self._known_distance_idx + 1,  # stan indexing
            r_nidx=self._unknown_distance_idx + 1,  # stan indexing
            log_flux_obs=np.log10(self._flux_selected),
            flux_obs=self._flux_selected,
            flux_sigma=self._flux_sigma,
            z_max=self._r_max,
            r_max=self._r_max,
            N_model=self._n_model,
        )

        # now append the model spaces
        for k, v in self._model_spaces.items():

            output[k] = v

        for k, v in self._auxiliary_quantities.items():

            output["%s_obs" % k] = v["obs_values"][self._selection]

        for k, v in output.items():

            if isinstance(v, np.int64):

                output[k] = int(v)

        return output

    def writeto(self, file_name: str) -> None:
        """
        Write population to an HDF5 file

        :param file_name: Name of the file
        :type file_name: str
        """

        with h5py.File(file_name, "w") as f:

            self._writeto(f)

    def addto(self, file_name: str, group_name: str) -> None:
        """
        Write population to a group in an existing
        HDF5 file.

        :param file_name: Name of the file
        :type file_name: str
        :param group_name: Name of the group
        :type group_name: str
        """

        with h5py.File(file_name, "r+") as f:

            g = f.create_group(group_name)

            self._writeto(g)

    def _writeto(self, f):
        """
        Write to an HDF5 file or group.

        :param f: h5py file or group handle
        """

        spatial_grp = f.create_group("spatial_params")

        for k, v in self._spatial_params.items():

            spatial_grp.create_dataset(k,
                                       data=np.array([v]),
                                       compression="lzf")

        if self._lf_params is not None:

            lum_grp = f.create_group("lf_params")

            for k, v in self._lf_params.items():

                lum_grp.create_dataset(k,
                                       data=np.array([v]),
                                       compression="lzf")

            f.attrs["lf_form"] = np.string_(self._lf_form)

            f.attrs["has_lf"] = True

        else:
            f.attrs["has_lf"] = False

        f.attrs["name"] = np.string_(self._name)
        f.attrs["spatial_form"] = np.string_(self._spatial_form)
        f.attrs["flux_sigma"] = self._flux_sigma
        f.attrs["n_model"] = self._n_model
        f.attrs["r_max"] = self._r_max
        f.attrs["seed"] = int(self._seed)

        f.create_dataset("luminosities",
                         data=self._luminosities,
                         compression="lzf")
        f.create_dataset("distances", data=self._distances, compression="lzf")
        f.create_dataset("known_distances",
                         data=self._known_distances,
                         compression="lzf")
        f.create_dataset("known_distance_idx",
                         data=self._known_distance_idx,
                         compression="lzf")
        f.create_dataset(
            "unknown_distance_idx",
            data=self._unknown_distance_idx,
            compression="lzf",
        )
        f.create_dataset("fluxes", data=self._fluxes, compression="lzf")
        f.create_dataset("flux_obs", data=self._flux_obs, compression="lzf")
        f.create_dataset("selection", data=self._selection, compression="lzf")
        f.create_dataset("theta", data=self._theta, compression="lzf")
        f.create_dataset("phi", data=self._phi, compression="lzf")

        aux_grp = f.create_group("auxiliary_quantities")

        for k, v in self._auxiliary_quantities.items():

            q_grp = aux_grp.create_group(k)
            q_grp.create_dataset("true_values",
                                 data=v["true_values"],
                                 compression="lzf")
            q_grp.create_dataset("obs_values",
                                 data=v["obs_values"],
                                 compression="lzf")

        model_grp = f.create_group("model_spaces")

        for k, v in self._model_spaces.items():

            model_grp.create_dataset(k, data=v, compression="lzf")

        # now store the truths
        recursively_save_dict_contents_to_group(f, "truth", self._truth)

        recursively_save_dict_contents_to_group(
            f, "graph", fill_graph_dict(nx.to_dict_of_dicts(self._graph)))

        # now store the popsynth
        recursively_save_dict_contents_to_group(f, "popsynth", self._pop_synth)

    @classmethod
    def from_file(cls, file_name: str):
        """
        Load a population from a file.

        :param file_name: Name of the file
        :type file_name: str
        """

        with h5py.File(file_name, "r") as f:

            return cls._loadfrom(f)

    @classmethod
    def from_group(cls, file_name: str, group_name: str):
        """
        Load a population from a group
        in a file.

        :param file_name: Name of the file
        :type file_name: str
        :param group_name: Name of the group
        :type group_name: str
        """

        with h5py.File(file_name, "r") as f:

            g = f[group_name]

            return cls._loadfrom(g)

    @classmethod
    def _loadfrom(cls, f):
        """
        Load from and HDF5 file or group.

        :param f: h5py file or group handle
        """

        spatial_params = {}

        for key in f["spatial_params"].keys():

            spatial_params[key] = f["spatial_params"][key][()][0]

        # we must double check that there are LF params

        try:

            if f.attrs["has_lf"]:
                lf_params = {}
                for key in f["lf_params"].keys():

                    lf_params[key] = f["lf_params"][key][()][0]
                lf_form = str(f.attrs["lf_form"])

            else:

                lf_params = None
                lf_form = None
        except:

            lf_params = None
            lf_form = None

        flux_sigma = f.attrs["flux_sigma"]
        n_model = f.attrs["n_model"]
        r_max = f.attrs["r_max"]
        seed = int(f.attrs["seed"])
        name = f.attrs["name"]

        spatial_form = str(f.attrs["spatial_form"])

        luminosities = f["luminosities"][()]
        distances = f["distances"][()]
        theta = f["theta"][()]
        phi = f["phi"][()]

        # right now this is just for older pops
        try:
            known_distances = f["known_distances"][()]
            known_distance_idx = (f["known_distance_idx"][()]).astype(int)
            unknown_distance_idx = (f["unknown_distance_idx"][()]).astype(int)

        except:

            known_distances = None
            known_distance_idx = None
            unknown_distance_idx = None

        fluxes = f["fluxes"][()]
        flux_obs = f["flux_obs"][()]
        selection = f["selection"][()]

        model_spaces = {}

        for k in f["model_spaces"].keys():

            model_spaces[str(k)] = f["model_spaces"][k][()]

        auxiliary_quantities = {}

        for k in f["auxiliary_quantities"].keys():

            auxiliary_quantities[str(k)] = {
                "true_values": f["auxiliary_quantities"][k]["true_values"][()],
                "obs_values": f["auxiliary_quantities"][k]["obs_values"][()],
            }

        truth = recursively_load_dict_contents_from_group(f, "truth")

        graph = nx.from_dict_of_dicts(
            clean_graph_dict(
                recursively_load_dict_contents_from_group(f, "graph")))

        pop_synth = recursively_load_dict_contents_from_group(f, "popsynth")

        return cls(
            luminosities=luminosities,
            distances=distances,
            known_distances=known_distances,
            known_distance_idx=known_distance_idx,
            unknown_distance_idx=unknown_distance_idx,
            fluxes=fluxes,
            flux_obs=flux_obs,
            selection=selection,
            flux_sigma=flux_sigma,
            n_model=n_model,
            r_max=r_max,
            lf_params=lf_params,
            spatial_params=spatial_params,
            model_spaces=model_spaces,
            seed=seed,
            name=name,
            spatial_form=spatial_form,
            lf_form=lf_form,
            auxiliary_quantities=auxiliary_quantities,
            truth=truth,
            graph=graph,
            theta=theta,
            phi=phi,
            pop_synth=pop_synth,
        )

    def to_sub_population(self, observed: bool = True) -> "Population":
        """
        Create a population that is down selected from either the
        observed or unobserved population

        :param observed: Extract the observed or unobserved object
        :type observed: bool
        :returns: A new population object
        :rtype: :class:`Population`
        """

        if observed:
            selection = self._selection

        else:

            selection = ~self._selection

        if self._auxiliary_quantities is not None:

            new_aux = {}

            for k, v in self._auxiliary_quantities.items():

                new_aux[k] = {
                    "true_values": v["true_values"][selection],
                    "obs_values": v["obs_values"][selection],
                }

        else:

            new_aux = None

        itr = 0
        known_distances = []
        known_distance_idx = []
        unknown_distance_idx = []

        for i, s in enumerate(selection):

            if s:

                if i in self._known_distances:

                    known_distances.append(self._distances[i])
                    known_distance_idx.append(itr)

                else:

                    unknown_distance_idx.append(itr)

                itr += 1

        return Population(
            luminosities=self._luminosities[selection],
            distances=self._distances[selection],
            known_distances=np.array(known_distances),
            known_distance_idx=np.array(known_distance_idx),
            unknown_distance_idx=np.array(unknown_distance_idx),
            fluxes=self._fluxes[selection],
            flux_obs=self._flux_obs[selection],
            selection=np.ones(sum(selection), dtype=bool),
            flux_sigma=self._flux_sigma,
            n_model=self._n_model,
            r_max=self._r_max,
            lf_params=self._lf_params,
            spatial_params=self._spatial_params,
            model_spaces=self._model_spaces,
            seed=self._seed,
            name=self._name,
            spatial_form=self._spatial_form,
            lf_form=self._lf_form,
            auxiliary_quantities=new_aux,
            truth=self._truth,
            graph=self._graph,
            theta=self._theta[selection],
            phi=self._phi[selection],
        )

    def display(self):
        """
        Display the simulation parameters.
        """

        info = "### %s simulation\nDetected %d out of %d objects" % (
            self._name,
            sum(self._selection),
            len(self._fluxes),
        )

        display(Markdown(info))

        if self._lf_params is not None:

            out = {"parameter": [], "value": []}

            display(Markdown("## Luminosity Function"))
            for k, v in self._lf_params.items():

                out["parameter"].append(k)
                out["value"].append(v)

            display(Math(self._lf_form))
            display(pd.DataFrame(out))
        out = {"parameter": [], "value": []}

        display(Markdown("## Spatial Function"))

        for k, v in self._spatial_params.items():

            out["parameter"].append(k)
            out["value"].append(v)

        display(Math(self._spatial_form))
        display(pd.DataFrame(out))

    def display_true_fluxes(self, ax=None, flux_color=dark, **kwargs):
        """
        Display the fluxes.

        :param ax: Axis on which to plot
        :param flux_color: Color of fluxes
        """

        if ax is None:
            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        ax.scatter(
            self._distances,
            self._fluxes,
            alpha=0.5,
            color=flux_color,
            edgecolors="none",
            label="True flux",
            **kwargs,
        )

        #        ax.axhline(self._boundary, color="grey", zorder=-5000, ls="--")

        # ax.set_xscale('log')
        ax.set_yscale("log")

        try:

            ax.set_ylim(
                bottom=min([self._fluxes.min(),
                            self._flux_selected.min()]))

        except:

            ax.set_ylim(bottom=self._fluxes.min())

        ax.set_xlabel("distance")
        ax.set_ylabel("flux")

    def display_obs_fluxes(self, ax=None, flux_color=dark, **kwargs):
        """
        Display the observed fluxes.

        :param ax: Axis on which to plot
        :param flux_color: Color of fluxes
        """

        # do not try to plot if there is nothing
        # to plot

        if self._no_detection:
            log.warning("There are no detections to display")
            if ax is not None:

                fig = ax.get_figure()

                return fig
            else:

                return

        if ax is None:
            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        ax.scatter(
            self._distance_selected,
            self._flux_selected,
            alpha=0.8,
            color=flux_color,
            edgecolors="none",
            label="Detected flux",
            **kwargs,
        )

        #        ax.axhline(self._boundary, color="grey", zorder=-5000, ls="--")
        # ax.set_xscale('log')
        ax.set_yscale("log")

        ax.set_ylim(bottom=min([self._fluxes.min(),
                                self._flux_selected.min()]))
        ax.set_xlim(right=self._r_max)

        ax.set_xlabel("distance")
        ax.set_ylabel("flux")
        return fig

    def display_fluxes(
        self,
        ax=None,
        true_color=light,
        obs_color=dark,
        arrow_color="k",
        with_arrows=True,
        **kwargs,
    ):
        """
        Display the fluxes.

        :param ax: Axis on which to plot
        :param true_color: Color of true values
        :param obs_color: Color of obs values
        :param arrow_color: Color of arrows
        :param with_arrows: If `True`, display arrows
        """

        if ax is None:
            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        self.display_true_fluxes(ax=ax, flux_color=true_color, **kwargs)

        if not self._no_detection:
            self.display_obs_fluxes(ax=ax, flux_color=obs_color, **kwargs)

        if (with_arrows) and (not self._no_detection):
            for start, stop, z in zip(
                    self._fluxes[self._selection],
                    self._flux_selected,
                    self._distance_selected,
            ):

                x = z
                y = start
                dx = 0
                dy = stop - start

                ax.arrow(
                    x,
                    y,
                    dx,
                    dy,
                    color=arrow_color,
                    head_width=0.05,
                    head_length=0.2 * np.abs(dy),
                    length_includes_head=True,
                )

        return fig

    def display_luminosities(
        self,
        ax=None,
        true_color=light,
        obs_color=dark,
        **kwargs,
    ):
        """
        Display the luminosities

        :param ax: Axis on which to plot
        :param true_color: Color of true values
        :param obs_color: Color of obs values
        """

        if ax is None:
            fig, ax = plt.subplots()

        else:

            fig = ax.get_figure()

        ax.scatter(self._distance_selected,
                   self._luminosity_selected,
                   s=5,
                   color=obs_color)
        ax.scatter(self._distance_hidden,
                   self._luminosity_hidden,
                   s=5,
                   color=true_color)

        return fig

    def _display_sphere(
        self,
        fluxes,
        distances,
        theta,
        phi,
        cmap="magma",
        distance_transform=None,
        use_log=False,
        fig=None,
        background_color="white",
        show=True,
        **kwargs,
    ):

        if len(fluxes) == 0:
            log.warning("There are no detections to display")

            return

        if fig is None:

            fig = ipv.figure()

        ipv.pylab.style.box_off()
        ipv.pylab.style.axes_off()
        ipv.pylab.style.set_style_dark()
        ipv.pylab.style.background_color(background_color)

        if distance_transform is not None:

            distance = distance_transform(distances)

        else:

            distance = distances

        x, y, z = _create_sphere_variables(self._r_max, distance, theta, phi)

        _, colors = array_to_cmap(fluxes, cmap, use_log=True)

        ipv.scatter(x, y, z, color=colors, marker="sphere", **kwargs)

        r_value = fig

        if show:

            ipv.xyzlim(self._r_max)
            fig.camera.up = [1, 0, 0]
            control = pythreejs.OrbitControls(controlling=fig.camera)
            fig.controls = control
            control.autoRotate = True
            fig.render_continuous = True
            # control.autoRotate = True
            # toggle_rotate = widgets.ToggleButton(description="Rotate")
            # widgets.jslink((control, "autoRotate"), (toggle_rotate, "value"))
            # r_value = toggle_rotate

        return fig

    def display_obs_fluxes_sphere(
        self,
        cmap="magma",
        distance_transform=None,
        use_log=False,
        background_color="white",
        show=True,
        **kwargs,
    ):

        theta = self._theta[self._selection]
        phi = self._phi[self._selection]

        fig = self._display_sphere(
            self._flux_selected,
            self._distance_selected,
            theta=theta,
            phi=phi,
            cmap=cmap,
            distance_transform=distance_transform,
            background_color=background_color,
            use_log=use_log,
            show=show,
            **kwargs,
        )

        if show:
            ipv.show()

        return fig

    def display_hidden_fluxes_sphere(
        self,
        cmap="magma",
        distance_transform=None,
        use_log=False,
        background_color="white",
        show=True,
        **kwargs,
    ):

        theta = self._theta[~self._selection]
        phi = self._phi[~self._selection]

        fig = self._display_sphere(
            self._flux_hidden,
            self._distance_hidden,
            theta=theta,
            phi=phi,
            cmap=cmap,
            distance_transform=distance_transform,
            background_color=background_color,
            use_log=use_log,
            show=show,
            **kwargs,
        )

        if show:
            ipv.show()

        return fig

    def display_flux_sphere(
        self,
        seen_cmap="Reds",
        unseen_cmap="Blues",
        distance_transform=None,
        use_log=False,
        background_color="white",
        **kwargs,
    ):

        fig = self.display_obs_fluxes_sphere(
            cmap=seen_cmap,
            distance_transform=distance_transform,
            use_log=use_log,
            background_color=background_color,
            show=False,
            **kwargs,
        )

        fig = self.display_hidden_fluxes_sphere(
            cmap=unseen_cmap,
            distance_transform=distance_transform,
            use_log=use_log,
            background_color=background_color,
            fig=fig,
            show=True,
            **kwargs,
        )

        return fig

    def display_distances(self, ax=None):
        """
        Display the distances

        :param ax: Axis on which to plot
        """

        if ax is None:
            fig, ax = plt.subplots()

        else:
            fig = ax.get_figure()

        bins = np.linspace(0, self._r_max, 40)

        ax.hist(
            self._distances,
            bins=bins,
            fc=dark,
            ec=dark_highlight,
            lw=1.5,
            label="Total Pop.",
        )
        ax.hist(
            self._distance_selected,
            bins=bins,
            #            facecolor=blue,
            #            edgecolor=blue_highlight,
            lw=1.5,
            alpha=1,
            label="Obs. Pop.",
        )

        ax.set_xlabel("z")
        ax.legend()
        # sns.despine(offset=5, trim=True);


def _create_sphere_variables(R, distance, theta, phi):
    x, y, z = xyz(distance, theta, phi)

    # u = np.linspace(0, 2 * np.pi, 100)
    # v = np.linspace(0, np.pi, 100)
    # x2 = R * np.outer(np.cos(u), np.sin(v))
    # y2 = R * np.outer(np.sin(u), np.sin(v))
    # z2 = R * np.outer(np.ones(np.size(u)), np.cos(v))

    return x, y, z
