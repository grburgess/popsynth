---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.8.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Quick start

A simple example of simulating a population via the built-in populations provided.

```python
%matplotlib notebook


import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(context="notebook", fscale=1, grid=False)
green = "#1DEBA6"
red = "#FF0059"
yellow = "#F6EF5B"


import popsynth

popsynth.update_logging_level("INFO")

import networkx as nx
import numpy as np
import warnings

warnings.simplefilter("ignore")
```

## A spherically homogenous population with a pareto luminosity function

**popsynth** comes with several types of populations preloaded. To create a population synthesizer, one simply instantiates the population form the **popsynth.populations** module.

```python
homo_pareto_synth = popsynth.populations.ParetoHomogeneousSphericalPopulation(
    Lambda=5, Lmin=1, alpha=2.0  # the density normalization  # lower bound on the LF
)  # index of the LF
homo_pareto_synth.display()
```


```python
# we can also display a graph of the object


options = {"node_color": green, "node_size": 2000, "width": 0.5}

pos = nx.drawing.nx_agraph.graphviz_layout(homo_pareto_synth.graph, prog="dot")

nx.draw(homo_pareto_synth.graph, with_labels=True, pos=pos, **options)
```


## Creating a survey


We can now sample from this population with the **draw_survey** function, but fits we need specfiy how the flux is selected

```python
flux_selector = popsynth.HardFluxSelection()
flux_selector.boundary = 1e-2

homo_pareto_synth.set_flux_selection(flux_selector)
```


```python
population = homo_pareto_synth.draw_survey(flux_sigma=0.1)
```

We now have created a population. How did we get here?

* Once the spatial and luminosity functions are specified, we can integrate out to a given distance and compute the number of expected objects.

* A Poisson draw with this mean is made to determine the number of total objects in the survey.

* Next all quantities are sampled (distance, luminosity)

* If needed, the luminosity is converted to a flux with a given observational error

* The selection function (in this case a hard cutoff) is applied

* A population object is created

We could have specified a soft cutoff (an inverse logit) with logarithmic with as well:

```python
homo_pareto_synth.clean()
flux_selector = popsynth.SoftFluxSelection()
flux_selector.boundary = 1e-2
flux_selector.strength = 10


homo_pareto_synth.set_flux_selection(flux_selector)

population = homo_pareto_synth.draw_survey(flux_sigma=0.1)
```

## The Population Object

The population object stores all the information about the sampled survey. This includes information on the latent parameters, measured parameters, and distances for both the selected and non-selected objects.


We can have a look at the flux-distance distribution from the survey. Here, pink dots are the *latent* flux value, i.e., without observational noise, and green dots are the *measured values for the *selected* objects. Arrows point from the latent to measured values.

```python
population.display_fluxes(obs_color=green, true_color=red)
```

For fun, we can display the fluxes on in a simulated universe in 3D

```python
fig = population.display_obs_fluxes_sphere()
```

The population object stores a lot of information. For example, an array of selection booleans:

```python
population.selection
```

We can retrieve selected and non-selected distances:

```python
population.selected_distances
```

```python
population.hidden_distances
```

## Saving the population
We can record the results of a population synth to an HDF5 file that maintains all the information from the run. The true values of the population parameters are always stored in the truth dictionary:


```python
population.truth
```

```python
population.writeto("saved_pop.h5")
```

```python
reloaded_population = popsynth.Population.from_file("saved_pop.h5")
```

```python
reloaded_population.truth
```

```python

```

## Creating populations from YAML files (experimental)

It is sometimes easier to quickly write down population in a YAML file without having to create all the objects in python. Let's a take a look at the format:

```yaml

# the seed
seed: 1234

# specifiy the luminosity distribution
# and it's parmeters
luminosity distribution:
    ParetoDistribution:
        Lmin: 1e51
        alpha: 2

# specifiy the flux selection function
# and it's parmeters
flux selection:
    HardFluxSelection:
        boundary: 1e-6

# specifiy the spatial distribution
# and it's parmeters

spatial distribution:
    ZPowerCosmoDistribution:
        Lambda: .5
        delta: -2
        r_max: 5

# specify the distance selection function
# and it's parmeters
distance selection:
    BernoulliSelection:
        probability: 0.5

# a spatial selection if needed
spatial selection:
    # None


# all the auxiliary functions
# these must be known to the
# registry at run time if
# the are custom!

auxiliary samplers:
    NormalAuxSampler:
        name: stellar_mass
        observed: False
        mu: 0
        sigma: 1
        selection:
        secondary:
        init variables:

    DemoSampler:
        name: demo
        observed: False
        selection:
            UpperBound:
                boundary: 20

    DemoSampler2:
        name: demo2
        observed: True
        selection:
        secondary: [demo, stellar_mass] # other samplers that this sampler depends on


```
We can load this yaml file into a population synth like this:







### Create our demo auxiliary samplers
read ahead int he docs for more details on auxiliary samplers

```python
class DemoSampler(popsynth.AuxiliarySampler):
    _auxiliary_sampler_name = "DemoSampler"
    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=2)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)

    def __init__(self):

        super(DemoSampler, self).__init__("demo", observed=False)

    def true_sampler(self, size):

        self._true_values = np.random.normal(self.mu, self.tau, size=size)


class DemoSampler2(popsynth.DerivedLumAuxSampler):
    _auxiliary_sampler_name = "DemoSampler2"
    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=2)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)
    sigma = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)

    def __init__(self):

        super(DemoSampler2, self).__init__("demo2")

    def true_sampler(self, size):

        secondary = self._secondary_samplers["demo"]

        self._true_values = (
            (np.random.normal(self.mu, self.tau, size=size))
            + secondary.true_values
            - np.log10(1 + self._distance)
        )

    def observation_sampler(self, size):

        self._obs_values = self._true_values + np.random.normal(
            0, self.sigma, size=size
        )

    def compute_luminosity(self):

        secondary = self._secondary_samplers["demo"]

        return (10 ** (self._true_values + 54)) / secondary.true_values
```

### Load the file
We use a saved file to demonstrate

```python
my_file = popsynth.utils.package_data.get_path_of_data_file("pop.yml")

ps = popsynth.PopulationSynth.from_file(my_file)
```

```python
options = {"node_color": green, "node_size": 2000, "width": 0.5}

pos = nx.drawing.nx_agraph.graphviz_layout(ps.graph, prog="dot")

nx.draw(ps.graph, with_labels=True, pos=pos, **options)
```


<!-- #region -->
We can see that our population was created correctly for us.


Now, this means we can easily pass populations around to our collaborators for testing
<!-- #endregion -->

```python
pop = ps.draw_survey(flux_sigma=0.5)
```

Now, since we can read the population synth from a file, we can also write one we have created with classes to a file:

```python
ps.to_dict()
```

```python
ps.write_to("/tmp/my_pop_synth.yml")
```

but our population synth is also serialized to our population!

```python
pop.pop_synth
```

Therefore we always know exactly how we simulated our data.

```python

```
