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
    display_name: Python3
    language: python
    name: Python3
---

# Auxiliary Samplers

Along with sampling the spatial and luminosity distributions, auxiliary properties and be sampled that both depend on and/or influence the luminosity. 


```python

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
%matplotlib notebook
from jupyterthemes import jtplot
jtplot.style(context='notebook', fscale=1, grid=False)
green = "#1DEBA6"
red = "#FF0059"
yellow = "#F6EF5B"

import warnings
warnings.simplefilter('ignore')


import popsynth
```

## Creating an auxiliary sampler
Let's create two auxiliary samplers that sample values from normal distributions with some dependency on each other.

First, we specify the main population. This time, we will chose a SFR-like redshift distribution and a Schecter luminosity function


```python
pop_gen = popsynth.populations.SchechterSFRPopulation(r0=10, 
                                                      rise=1.,
                                                      decay=1.,
                                                      peak=1.,
                                                      Lmin=1E50,
                                                      alpha=2.)
```

<!-- #region -->
Suppose we have a property "demo" that we want to sample as well. For this property, we do not observe it directly. We will get to that. 


We create an AuxiliarySampler child class, and define the *true_sampler* for the latent values
<!-- #endregion -->

```python
class DemoSampler(popsynth.AuxiliarySampler):
    
    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=2)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)
    
    
    def __init__(self):
 
        # pass up to the super class
        super(DemoSampler, self).__init__('demo', observed=False)
        
    def true_sampler(self, size):
        
        # sample the latent values for this property
        
        self._true_values =  np.random.normal(self.mu, self.tau, size=size)
        
```

Now we instantiate it and then assign it our pop_gen object. Then we draw out survey

```python
demo1 = DemoSampler()

pop_gen.add_observed_quantity(demo1)

population = pop_gen.draw_survey(boundary=1E-8, hard_cut=True, flux_sigma= 0.1,verbose=True)

options = {
'node_color':green,
'node_size': 2000,
'width': .5}


pos=nx.drawing.nx_agraph.graphviz_layout(
        population.graph, prog='dot'
    )
    
nx.draw(population.graph, with_labels=True,pos=pos, **options)

```

```python
population.display_fluxes(obs_color=green, true_color=red,s=15);
```

We can see that the population has stored out demo auxiliary property

```python
population.demo
```

```python
population.demo_selected
```

## Observed auxiliary properties and dependent parameters

Suppose now we want to simulate a property that is observed by an instrument but depends on latent parameters.

We will create a second demo sampler and tell it what the observational error is as well as how to read from a secondary sampler:

```python
class DemoSampler2(popsynth.AuxiliarySampler):
    
    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=2)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)
    sigma = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)
    
    
    
    def __init__(self,):

        # this time set observed=True
        super(DemoSampler2, self).__init__('demo2', observed=True, uses_distance=True)
        
    def true_sampler(self, size):
        
        # we access the secondary sampler dictionary. In this 
        # case "demo". This itself is a sampler with 
        # <>.true_values as a parameter
        secondary = self._secondary_samplers['demo']
        
        # now we sample the demo2 latent values and add on the dependence of "demo"
        
        tmp =  (np.random.normal(self.mu , self.tau, size=size))
        
        # for fun, we can substract the log of the distance as all
        # auxiliary samples know about their distances
        
        self._true_values = tmp + secondary.true_values - np.log10(1+self._distance)
        
    def observation_sampler(self, size):
        
        # here we define the "observed" values, i.e., the latened values 
        # with observational error
        
        self._obs_values =  self._true_values + np.random.normal(0, self.sigma, size=size)

```

We recreate our base sampler:

```python
pop_gen = popsynth.populations.SchechterSFRPopulation(r0=10, 
                                                      rise=1.,
                                                      decay=1.,
                                                      peak=1.,
                                                      Lmin=1E50,
                                                      alpha=2.)

```

Now, make a new *demo1*, but this time we do not have to attach it to the base sampler. Instead, we will assign it as a secondary sampler to *demo2* and **popsynth** is smart enough to search for it when it draws a survey. 

```python
demo1 = DemoSampler()


demo2 = DemoSampler2()

demo2.set_secondary_sampler(demo1)

# attach to the base sampler
pop_gen.add_observed_quantity(demo2)



```

```python
pos=nx.drawing.nx_agraph.graphviz_layout(
        pop_gen.graph, prog='dot'
    )


fig, ax = plt.subplots()
    
    
nx.draw(pop_gen.graph, with_labels=True,pos=pos,ax=ax, **options)


```

```python
population = pop_gen.draw_survey(boundary=1E-8, hard_cut=True, flux_sigma= 0.1,verbose=True)
```

```python
fig, ax = plt.subplots()

ax.scatter(population.demo2_selected, population.demo_selected ,c=green,s=40)

ax.scatter(population.demo2, population.demo ,c=red,s=20)
```

## Derived Luminosity sampler

Sometimes, the luminosity does not come directly from a distribution. Rather, it is computed from other quantities. In these cases, we want to use the **DerivedLumAuxSampler** class.

This allows you to sample auxiliary parameters and compute a luminosity from those. 

```python

class DemoSampler3(popsynth.DerivedLumAuxSampler):
    
    mu = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1)
    tau = popsynth.auxiliary_sampler.AuxiliaryParameter(default=1, vmin=0)
    
    def __init__(self, mu=2, tau=1., sigma=1):
    
        # this time set observed=True
        super(DemoSampler3, self).__init__('demo3', uses_distance=False)
        
    def true_sampler(self, size):
    
        # draw a random number
        tmp =  np.random.normal(self.mu , self.tau, size=size)
     
        self._true_values = tmp 
        
    def compute_luminosity(self):
        
        # compute the luminosity
        secondary = self._secondary_samplers["demo"]

        return (10 ** (self._true_values + 54)) + secondary.true_values
```

```python
pop_gen = popsynth.populations.SchechterSFRPopulation(r0=10, 
                                                      rise=1.,
                                                      decay=1.,
                                                      peak=1.,
                                                      Lmin=1E50,
                                                      alpha=2.)




```

```python
demo1 = DemoSampler()


demo3 = DemoSampler3()

demo3.set_secondary_sampler(demo1)

# attach to the base sampler
pop_gen.add_observed_quantity(demo3)


pos=nx.drawing.nx_agraph.graphviz_layout(
        pop_gen.graph, prog='dot'
    )
 
fig, ax = plt.subplots()
    
nx.draw(pop_gen.graph, with_labels=True,pos=pos, **options, ax=ax)
```

```python
population = pop_gen.draw_survey(boundary=1E-5, hard_cut=True, flux_sigma= 0.1,verbose=True)
```

```python
population.display_fluxes(obs_color=green, true_color=red,s=15);
```

```python

```
