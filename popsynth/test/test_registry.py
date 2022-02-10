from popsynth import (
    list_available_auxiliary_samplers,
    list_available_distributions,
    list_available_selection_functions,
)


def test_listings():

    list_available_auxiliary_samplers()
    list_available_distributions()
    list_available_selection_functions()
