from class_registry import ClassRegistry

auxiliary_parameter_registry = ClassRegistry("_auxiliary_sampler_name")

distribution_registry = ClassRegistry("_distribution_name")

selection_registry = ClassRegistry("_selection_name")


def list_available_distributions():
    for k, v in distribution_registry.items():

        print(k)


def list_available_selection_functions():
    for k, v in selection_registry.items():

        print(k)


def list_available_auxiliary_samplers():
    for k, v in auxiliary_parameter_registry.items():

        print(k)


__all__ = [
    "auxiliary_parameter_registry",
    "distribution_registry",
    "distribution_registry",
]
