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
jtplot.style(context='notebook', fscale=1, grid=False)
green = "#1DEBA6"
red = "#FF0059"
yellow = "#F6EF5B"


import popsynth
popsynth.update_logging_level("INFO")

import networkx as nx

import warnings
warnings.simplefilter('ignore')
```

## A spherically homogenous population with a pareto luminosity function

**popsynth** comes with several types of populations preloaded. To create a population synthesizer, one simply instantiates the population form the **popsynth.populations** module.

```python
homo_pareto_synth = popsynth.populations.ParetoHomogeneousSphericalPopulation(Lambda=0.25, # the density normalization
                                                                              Lmin=1, # lower bound on the LF
                                                                              alpha=2.) # index of the LF
homo_pareto_synth.display()

```

```python
# we can also display a graph of the object


options = {
'node_color':green,
'node_size': 2000,
'width': .5}

pos=nx.drawing.nx_agraph.graphviz_layout(
        homo_pareto_synth.graph, prog='dot'
    )
    
nx.draw(homo_pareto_synth.graph, with_labels=True,pos=pos, **options)


```

## Creating a survey


We can now sample from this population with the **draw_survey** function, but fits we need specfiy how the flux is selected

```python
flux_selector = popsynth.HardFluxSelection()
flux_selector.boundary=1E-2

homo_pareto_synth.set_flux_selection(flux_selector)

```

```python
population = homo_pareto_synth.draw_survey(
    flux_sigma= 0.1)
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
flux_selector.boundary=1E-2
flux_selector.strength=10


homo_pareto_synth.set_flux_selection(flux_selector)

population = homo_pareto_synth.draw_survey( flux_sigma= 0.1)
```

## The Population Object

The population object stores all the information about the sampled survey. This includes information on the latent parameters, measured parameters, and distances for both the selected and non-selected objects.


We can have a look at the flux-distance distribution from the survey. Here, pink dots are the *latent* flux value, i.e., without observational noise, and green dots are the *measured values for the *selected* objects. Arrows point from the latent to measured values. 

```python
population.display_fluxes(obs_color=green, true_color=red);
```

For fun, we can display the fluxes on in a simulated universe in 3D

```python
population.display_obs_fluxes_sphere();
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
population.writeto('saved_pop.h5')
```

```python
reloaded_population = popsynth.Population.from_file('saved_pop.h5')
```

```python
reloaded_population.truth
```

```python

```

## Creating populations from YAML files

```python

```
