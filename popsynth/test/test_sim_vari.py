import numpy as np
import numpy.testing as npt
import pytest

from popsynth.simulated_variable import SimulatedVariable

_obs = np.array([1, 2, 3])
_latent = np.array([5, 6, 7])
_selection = np.array([1, 0, 1], dtype=bool)

_latent_bad = np.array([5, 6])
_selection_bad = np.array([1, 0], dtype=bool)


def test_simulated_variable_constructor():

    sv = SimulatedVariable(_obs, _latent, _selection)

    npt.assert_array_equal(sv.view(np.ndarray), _obs)

    npt.assert_array_equal(sv.latent, _latent)

    npt.assert_array_equal(sv._selection, _selection)

    with pytest.raises(AssertionError):

        SimulatedVariable(_obs, _latent_bad, _selection)

    with pytest.raises(AssertionError):

        SimulatedVariable(_obs, _latent, _selection_bad)

    with pytest.raises(AssertionError):

        SimulatedVariable(_obs, _latent_bad, _selection_bad)

    sel = sv.selected

    npt.assert_array_equal(sel.view(np.ndarray), _obs[_selection])

    npt.assert_array_equal(sel.latent, _latent[_selection])

    npt.assert_array_equal(sel._selection, _selection[_selection])

    nsel = sv.non_selected

    npt.assert_array_equal(nsel.view(np.ndarray), _obs[~_selection])

    npt.assert_array_equal(nsel.latent, _latent[~_selection])

    npt.assert_array_equal(nsel._selection, _selection[~_selection])


def test_simulated_variable_math():

    lat2 = np.power(_latent, 2)
    latp1 = _latent + 1

    obs2 = np.power(_obs, 2)
    obsp1 = _obs + 1

    sv = SimulatedVariable(_obs, _latent, _selection)

    sq = np.power(sv, 2)

    npt.assert_array_equal(sq.view(np.ndarray), obs2)

    npt.assert_array_equal(sq.latent, lat2)

    sq = sv + 1

    npt.assert_array_equal(sq.view(np.ndarray), obsp1)

    npt.assert_array_equal(sq.latent, latp1)
