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
# Selections

Selections on parameters including flux, distance and any auxiliary
variables, can be performed in arbitrarily complex way.  We are
familiar now with how to add selections onto fluxes and distances, now
we will examine in more detail.


## built in selection functions

There are several available selection functions:

```python
import matplotlib.pyplot as plt
import numpy as np
%matplotlib notebook

import popsynth
popsynth.loud_mode()
popsynth.list_available_selection_functions()
```

We can use these to set selections on parameters. Let's add a dummy
parameter that is sampled from a normal distribution:

```python
aux_parameter = popsynth.NormalAuxSampler(name="dummy",observed=False)
aux_parameter.mu = 0
aux_parameter.sigma = 1


```

Now we will use the built in Box selection function. Here, we will
assign it to an auxiliary sampler, so we need to tell it to select on
the observed value:

```python
box_select = popsynth.BoxSelection(name="aux_selector", use_obs_value=True)
box_select.vmin=0
box_select.vmax=0.5
```

We can also add on a selection function for the flux

```python
flux_select = popsynth.HardFluxSelection()
flux_select.boundary=1e-6
```

Now, we can put it all together and create a survey:

```python
ps = popsynth.SchechterZPowerCosmoPopulation(Lambda=50, delta=-2, Lmin=1e52, alpha=1.5, seed=1234)

aux_parameter.set_selection_probability(box_select)

ps.set_flux_selection(flux_select)

ps.add_auxiliary_sampler(aux_parameter)

pop = ps.draw_survey()


```

```python
fig, ax = plt.subplots()

ax.scatter(np.log10(pop.fluxes_observed),pop.dummy,color="purple",alpha=0.7, label="total")
ax.scatter(np.log10(pop.selected_fluxes_observed ),pop.dummy_selected,color="yellow",alpha=0.7, label="selected")

ax.set(xlabel="log10 fluxes", ylabel="dummy")
ax.legend()
```

