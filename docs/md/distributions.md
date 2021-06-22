---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.11.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Core Concept
```popsynth``` core function is to create **observed** surveys from **latent** population models. 


First, let's define what a population of objects is in terms of a
generative model. The two main ingredients are the objects' spatial
distribution ($\lambda(\vec{r}; \vec{\psi})$) and the distribution of
their inherent properties ($\pi(\vec{\phi} | \vec{\psi})$). Here,
$\vec{\psi}$ are the latent population parameters, $\vec{r}$ are the
spatial locations of the objects, and $\vec{\phi}$ are the properties
of the individual objects (luminosity, spin, viewing angle, mass,
etc.). The spatial distribution is defined such that:

$$\frac{d \Lambda}{dt}(\vec{\psi}) = \int d r \frac{dV}{dr} \lambda(\vec{r}; \vec{\psi})$$

is the intensity of objects for a given set of population
parameters. With these definitions we can define the probability for
an object to have position $\vec{r}$ and properties $\vec{\phi}$ as

$$\pi(\vec{r}, \vec{\phi} | \vec{\psi}) = \frac{\lambda(\vec{r}; \vec{\psi})  \pi(\vec{\phi} | \vec{\psi})}{ \int d r \frac{dV}{dr} \lambda(\vec{r}; \vec{\psi})} $$

```popsynth``` allows you to specify these spatial and property
distributions in an object-oriented way to create surveys. The final
ingredient to creating a sample for a survey is knowing how many
objects to sample from the population (before any selection effects
are applied). Often, we see this number in simulation frameworks
presented as "we draw N objects to guarantee we have enough." This is
incorrect. A survey takes place over a given period of time ($\Delta
t$) in which observed objects are counted. This is a description of a
Poisson process. Thus, the number of objects in a simulation of this
survey is a draw from a Poisson distribution:

$$N \sim \mathrm{Poisson(\Delta t \frac{d\Lambda}{dt})} \mathrm{.}$$

Thus, ```popsynth``` first numerically integrates the spatial
distribution to determine the Poisson rate parameter for the given
$\vec{\psi}$, then makes a Poisson draw for the number of objects in
the population survey. For each object, positions and properties are
drawn with arbitrary dependencies between them. Finally, selection
functions are applied to either latent or observed (with or without
measurement error) properties.


**Note:** If instead we draw a preset number of objects, as is done in
many astrophysical population simulation frameworks, it is equivalent
to running a survey up until that specific number of objects is
detected. This process is distributed as a negative binomial process,
i.e, wait for a number of successes and requires a different
statistical framework to compare models to data.

In the following, the process for constructing distributions and
populations is described.


# Distributions


The basic required object to create a population synth are a spatial
and (optional if a derived luminosity sampler is create) luminosity
distribution.


```python
%matplotlib inline


import matplotlib.pyplot as plt
from jupyterthemes import jtplot

jtplot.style(context="notebook", fscale=1, grid=False)
purple = "#B833FF"
yellow = "#F6EF5B"

import networkx as nx
import numpy as np
import warnings

warnings.simplefilter("ignore")
```

`popsynth` comes with several built in distributions included

```python
import popsynth
popsynth.update_logging_level("INFO")


popsynth.list_available_distributions()
```

## Creating a simple population synth

First we create a spatial distribution, in the case, a Spherical distribution with a power law density.


```python
spatial_distribution = popsynth.ZPowerSphericalDistribution()

spatial_distribution.Lambda = 100
spatial_distribution.delta = -2
spatial_distribution.r_max = 10

```

And now we create a powerlaw luminosity distribution

```python
luminosity_distribution = popsynth.ParetoDistribution()

luminosity_distribution.alpha = 1.5
luminosity_distribution.Lmin = 1

```

Combining these together with a random seed, we have a population synthesis object

```python
pop_gen = popsynth.PopulationSynth(luminosity_distribution=luminosity_distribution, 
                                   spatial_distribution = spatial_distribution,
                                   seed=1234
                                  
                                  
                                  )
```

```python
pop_gen.display()
```

```python
population = pop_gen.draw_survey()
```

```python
fig=population.display_obs_fluxes_sphere(background_color="black",size=0.7);
```

## Cosmological Distributions

If we want to create cosmological spatial distributions, we can use some of those that are built in.


```python
spatial_distribution = popsynth.ZPowerCosmoDistribution()
spatial_distribution.Lambda = 100
spatial_distribution.delta = -2
spatial_distribution.r_max = 10

```

These distributions know about the cosmological Universe and have their fluxes computed using the luminosity distance rather than linear distace. 

```python
luminosity_distribution = popsynth.SchechterDistribution()

luminosity_distribution.alpha = 1.5
luminosity_distribution.Lmin = 1


```

```python
pop_gen = popsynth.PopulationSynth(luminosity_distribution=luminosity_distribution, 
                                   spatial_distribution = spatial_distribution,
                                   seed=1234
                                  
                                  
                                  )
```

```python
pop_gen.display()
```

```python
population = pop_gen.draw_survey()
```

```python
fig=population.display_obs_fluxes_sphere(cmap="viridis", background_color="black",size=0.7);
```

The cosmological parameters used when simulating are stored in the cosmology object:

```python
popsynth.cosmology.Om
```

```python
popsynth.cosmology.h0
```

```python
popsynth.cosmology.Ode
```


<div class="alert alert-info">

**Note:** The values of Om and h0 can be changed and will change the values of all cosmological calculations

</div>




```python
popsynth.cosmology.Om=0.7
```

```python
popsynth.cosmology.Ode
```

Let's re run the last simulation to see how this changes things

```python
pop_gen.clean()
```

```python
population = pop_gen.draw_survey()
```

```python
fig=population.display_obs_fluxes_sphere(background_color="black",size=0.7);
```

```python

```
