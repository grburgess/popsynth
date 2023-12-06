import numpy as np

from .utils.logging import setup_logger

log = setup_logger(__name__)


class SimulatedVariable(np.ndarray):
    """
    SimulatedVariables hold the observed and latent values
    from a population synth variable as well as its selection.

    The array displays as the observed values. The latent values
    can be accessed with x.latent

    Subsets of selected and non-selected values can be accessed:

    x.selected
    x.non_selected

    which return a SimulatedVariable that is a subset of the values.

    Math operations applied to the object will be applied to the observed
    and latent values.


    """

    def __new__(
        cls,
        observed_values: np.ndarray,
        latent_values: np.ndarray,
        selection: np.ndarray,
    ):
        # Input array is an already formed ndarray instance
        # We first cast to be our class type

        obj = np.asarray(observed_values).view(cls)

        if len(selection) != len(observed_values):

            log.error(
                f"selection ({len(selection)}) is not the same length as observation ({len(observed_values)})"
            )

            raise AssertionError()

        if len(latent_values) != len(observed_values):

            log.error("selection is not the same length as observation")

            raise AssertionError()

        # the selection on the value
        obj._selection: np.ndarray = selection

        # the latent values of the variable
        obj._latent_values: np.ndarray = latent_values

        # Finally, we must return the newly created object:
        return obj

    @property
    def latent(self) -> np.ndarray:
        """
        The latent values simulated
        which are unobscured by measurement
        error

        :returns:

        """

        return self._latent_values

    @property
    def selected(self) -> "SimulatedVariable":
        """
        returns the latent and observed values
        that were selected as a new SimulatedVariable
        """

        selected = self.view(np.ndarray)[self._selection]
        selected_latent = self._latent_values[self._selection]

        new_selection = np.ones_like(selected)

        return SimulatedVariable(selected, selected_latent, new_selection)

    @property
    def non_selected(self) -> "SimulatedVariable":
        """
        returns the latent and observed values
        that were selected as a new SimulatedVariable
        """
        non_selected = self.view(np.ndarray)[~self._selection]
        non_selected_latent = self._latent_values[~self._selection]

        new_selection = np.zeros_like(non_selected)

        return SimulatedVariable(non_selected, non_selected_latent,
                                 new_selection)

    def __array_finalize__(self, obj):

        # see InfoArray.__array_finalize__ for comments
        if obj is None:
            return

        # Add the value

        self._selection = getattr(obj, "_selection", None)

        self._latent_values = getattr(obj, "_latent_values", None)

    def __getitem__(self, item):

        obs = self.view(np.ndarray).__getitem__(item)

        if not isinstance(obs, np.ndarray):

            return obs

        latent = self._latent_values.__getitem__(item)

        selection = self._selection.__getitem__(item)

        return SimulatedVariable(obs, latent, selection)

    def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
        args = []
        in_no = []
        latent_args = []

        is_reduction = False

        for i, input_ in enumerate(inputs):
            if isinstance(input_, SimulatedVariable):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
                latent_args.append(input_._latent_values.view(np.ndarray))
            else:
                args.append(input_)
                latent_args.append(input_)

        outputs = out
        out_no = []
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, SimulatedVariable):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None, ) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no

        results = super().__array_ufunc__(ufunc, method, *args, **kwargs)

        latent_results = super().__array_ufunc__(ufunc, method, *latent_args,
                                                 **kwargs)

        if results is NotImplemented:
            return NotImplemented

        if latent_results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], SimulatedVariable):

                print("oh shit")

                pass
            #                inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results, )
            latent_results = (latent_results, )

        final_results = []

        for result, latent_result, output in zip(results, latent_results,
                                                 outputs):

            if output is None:

                if isinstance(result, np.ndarray):

                    out = SimulatedVariable(result, latent_result,
                                            self._selection)

                else:

                    out = result

            else:

                out = output

            final_results.append(out)

        results = tuple(final_results)

        if results and isinstance(results[0], SimulatedVariable):

            pass

        return results[0] if len(results) == 1 else results
