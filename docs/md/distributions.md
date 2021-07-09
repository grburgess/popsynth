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

spatial_distribution.Lambda = 30
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

If we want to create cosmological spatial distributions, we can use
some of those that are built in.

```python
spatial_distribution = popsynth.ZPowerCosmoDistribution()
spatial_distribution.Lambda = 100
spatial_distribution.delta = -2
spatial_distribution.r_max = 10

```

These distributions know about the cosmological Universe and have
their fluxes computed using the luminosity distance rather than linear
distace.

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
