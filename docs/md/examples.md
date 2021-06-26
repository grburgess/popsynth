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

# Examples

Here are some examples of astrophysical populations from publications.


## Short GRBS 

In [Ghirlanda et al. 2016](https://arxiv.org/abs/1607.07875) a fitting algorithm was used to determine the redshift and luminosity of short GRBS. We can use the parameters to reproduce the population and the observed GBM survey.

```python
from popsynth import SFRDistribution, BPLDistribution, PopulationSynth, NormalAuxSampler, AuxiliarySampler, HardFluxSelection
from popsynth import update_logging_level
update_logging_level("INFO")
```

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

In the work, the luminosity function of short GRBs is model as a broken power law.

```python
bpl = BPLDistribution()

bpl.alpha = -0.53
bpl.beta = -3.4
bpl.Lmin = 1e47 # erg/s
bpl.Lbreak = 2.8e52
bpl.Lmax = 1e55

```

To model the redshift distribution, an empirical form from [Cole et al 2001](https://academic.oup.com/mnras/article/326/1/255/1026734?login=true) is used. In ```popsynth``` we call this the ```SFRDistribution``` (but perhaps a better name is needed).

```python
sfr = SFRDistribution()
```

```python
sfr.r0 = 5.
sfr.a = 1
sfr.rise = 2.8
sfr.decay = 3.5
sfr.peak = 2.3
```

We can checkout how the rate changes with redshift

```python
fig, ax = plt.subplots()

z = np.linspace(0,5,100)

ax.plot(z, sfr.dNdV(z), color=purple)
ax.set_xlabel("z")
ax.set_ylabel(r"$\frac{\mathrm{d}N}{\mathrm{d}V}$")
```

<!-- #region -->
In their model, the authors also have some secondary parameters that are connected to the luminosity. These are the  parameters for the spectrum of the GRB. It is proposed that the spectra peak energy (Ep) is linked to the luminosity by a power law relation:


$$ \log E_{\mathrm{p}} \propto a + b \log L$$

We can build an auxiliary sample to simulate this as well. But we will also add a bit of scatter to the intercept of the relation.
<!-- #endregion -->

```python
intercept =NormalAuxSampler(name="intercept", observed=False)

intercept.mu = 0.034
intercept.sigma = .005



```

```python
class EpSampler(AuxiliarySampler):
    
    _auxiliary_sampler_name = "EpSampler"

    def __init__(self):

        # pass up to the super class
        super(EpSampler, self).__init__("Ep", observed=True, uses_luminosity = True)

    def true_sampler(self, size):

        # we will get the intercept's latent (true) value
        # from its sampler
        
        intercept = self._secondary_samplers["intercept"].true_values
        
        slope = 0.84

        self._true_values = np.power(10., intercept + slope * np.log10(self._luminosity/1e52) + np.log10(670.))
        
    def observation_sampler(self, size):
        
        # we will also add some measurement error to Ep
        self._obs_values = self._true_values + np.random.normal(0., 10, size=size)
        
```

Now we can put it all together.

```python
pop_synth = PopulationSynth(spatial_distribution=sfr, luminosity_distribution=bpl)
```

We will have a hard flux selection which is Fermi-GBM's fluz limit of ~ 1e-7 erg/s/cm2

```python
selection = HardFluxSelection()
selection.boundary = 1e-7
```

```python
pop_synth.set_flux_selection(selection)
```

We need to add the Ep sampler. Once we set the intercept sampler as a secondary it will automatically be added to the population synth.

```python
ep = EpSampler()
```

```python
ep.set_secondary_sampler(intercept)
```

```python
pop_synth.add_auxiliary_sampler(ep)
```

We are ready to sample our population. We will add some measurement uncertainty to the fluxes as well.

```python
population = pop_synth.draw_survey(flux_sigma=0.2)
```

```python
population.display_fluxes(true_color=purple, obs_color=yellow, with_arrows=False, s= 5);
```

Let's look at our distribution of Ep

```python
fig, ax = plt.subplots()

ax.hist(np.log10(population.Ep_obs[population.selection]), histtype="step", color=yellow, lw=3, label="Ep observed")
ax.hist(np.log10(population.Ep[~population.selection]), histtype="step", color=purple, lw=3,  label="Ep hidden")
ax.set_xlabel("log Ep")

ax.legend()
```

```python
fig, ax = plt.subplots()


ax.scatter(population.fluxes_observed[~population.selection],
           population.Ep_obs[~population.selection],c=purple, alpha=0.5)
ax.scatter(population.fluxes_observed[population.selection],
           population.Ep_obs[population.selection],c=yellow, alpha=0.5)

ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("log Ep")
ax.set_xlabel("log Flux")
```

Does this look like the observed catalogs?

## BL Lac blazars
A model for the luminosity function and cosmic evolution of BL Lac type blazars is presented in [Ajello et al. 2014](https://arxiv.org/abs/1310.0006), based on observations in gamma-rays with the Fermi-LAT instrument.

We can use the results of this paper to build a BL Lac population that is able to reproduce the results reported in the recent [4FGL Fermi-LAT catalog](https://arxiv.org/abs/1902.10045) reasonably well.

```python
from scipy import special as sf
from popsynth import (ZPowerCosmoDistribution, SoftFluxSelection,
                      GalacticPlaneSelection)
```

The work mentioned above presents 3 models for the BL Lac luminosity function. Here, we focus on the case of pure density evolution (referred to as PDE in the paper). In this case, the BL Lac population is parametrised as having a broken power law luminosity distribution, with an independent density evolution following a cosmological power law distribution.

We work with a luminosity range of $L_\mathrm{min} = 7\times 10^{43}$ erg $\mathrm{s}^{-1}$ and $L_\mathrm{max} = 10^{52}$ erg $\mathrm{s}^{-1}$ following Ajello et al. 2014. All luminosities are in units of erg $\mathrm{s}^{-1}$. Similarly, the maxium redshift considered in $z=6$.

We start by setting up the broken power law distribution (`BPLDistribution`).

```python
bpl = BPLDistribution()
bpl.alpha = -1.5
bpl.beta = -2.5
bpl.Lmin = 7e43
bpl.Lmax = 1e52
bpl.Lbreak = 1e47

fig, ax = plt.subplots()
L = np.geomspace(bpl.Lmin, bpl.Lmax)
ax.plot(L, bpl.phi(L), color=purple)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"L [erg $\mathrm{s}^{-1}$]")
```

We now move to the redshift distribution. Following the paper, we parametrize this as a negative power law in $z$. for the purpose of this example, we assume that Bl Lac blazars emit with a steady state. This means that we need to set the `is_rate` parameter to `False` when defining the `ZPowerCosmoDistribution` cosmological distribution. What we mean when we do this is that our local number density, `Lambda` is not per unit time. We also want to survey the whole sky, so we integrate over $4\pi$ sr in the value that we pass to the `Lambda`. 

```python
zpow = ZPowerCosmoDistribution()
zpow.Lambda = 9000 # Gpc^-3 sr 
zpow.delta = -6

fig, ax = plt.subplots()
z = np.linspace(0.01, 6)
ax.plot(z, zpow.dNdV(z), color=purple)
ax.set_yscale("log")
ax.set_xlabel("z")
ax.set_ylabel(r"$\frac{\mathrm{d}N}{\mathrm{d}V}$")
```

Apart from their redshifts and luminosities, BL Lacs also have other properties. As a simple example, we can consider their spectral index, assuming that the gamma-ray emission is well modelled in the energy range of interest (0.1 to 100 GeV) by a simple power law. 

We assume these true values of these indices are normally distributed with mean, $\mu$, and standard deviation, $\tau$. Additionally, we recognise that these are reconstructed quantities in real surveys, with uncertain values. This is reflected in the error, $\sigma$, on the observed values.

```python
index = NormalAuxSampler(name="index")
index.mu = 2.1
index.tau = 0.25
index.sigma = 0.1
```

We know that the Fermi-LAT detector cannot detect all objects in the Universe, and it is necessary to model some kind of selection function. In general, brighter and spectrally harder objects are easier to detect. We take this into acount by selecting on the flux, $F=L/4\pi d_L^2(z)$, where $d_L$ is the luminosity distance in cm.

This selection effect will not really be a hard boundary, although we could approximate it as such. In reality, the probability to detect an object increases as a function of its flux. To capture this effect, we consider a `SoftFluxSelection` as follows.

```python
flux_selector = SoftFluxSelection()
flux_selector.boundary = 4e-12 # erg cm^-2 s^-1
flux_selector.strength = 2

# This is what is happening under the hood
fig, ax = plt.subplots()
F = np.geomspace(1e-15, 1e-8)
ax.plot(F, sf.expit(flux_selector.strength * (np.log10(F) - np.log10(flux_selector.boundary))))
ax.set_xscale("log")
ax.set_xlabel("F [erg $\mathrm{cm}^{-2}$ $\mathrm{s}^{-1}$]")
ax.set_ylabel("Detection prob.")
```

Finally, sometimes it is harder to detect objects near the bright Galactic plane. We take this into account by excluding $10^\circ$ either side of the plane in Galactic longitude using the `GalacticPlaneSelector`.

```python
gp = GalacticPlaneSelection()
gp.b_limit = 10
```

Now, lets finally bring all this together to make a simulated population. Here, we defined our luminosity and spatial distributions already, so we can use them directly in `PopulationSynth`, but there is also the `BPLZPowerCosmoPopulation` available as a quick interface.

```python
# Main pop synth
pop_synth = PopulationSynth(spatial_distribution=zpow, luminosity_distribution=bpl)

# Add our selection effects
pop_synth.set_flux_selection(flux_selector)
pop_synth.add_spatial_selector(gp)

# Add our auxiliary param - spectral index
pop_synth.add_observed_quantity(index)
```

Lets run it! The last parameter to set is adding some uncertainty to our observed flux values.

```python
population = pop_synth.draw_survey(flux_sigma=0.1)
```

We can now have a look at the properties of this simulated population, such as the detected and undetected fluxes and distances.

```python
population.display_fluxes(true_color=purple, obs_color=yellow, with_arrows=False, s=5);
```

```python
fig, ax = plt.subplots()
ax.hist(population.distances, color=purple, histtype="step", lw=3, label="All")
ax.hist(population.distances[population.selection], histtype="step", lw=3, 
        color=yellow, label="Detected")
ax.set_xlabel("$z$")
ax.legend()
```

We can also check out the spectral index distribution.

```python
fig, ax = plt.subplots()
ax.hist(population.index, color=purple, histtype="step", lw=3, label="All")
ax.hist(population.index[population.selection], histtype="step", lw=3, 
        color=yellow, label="Detected")
ax.set_xlabel("Spectral index")
ax.legend()
```

We can now imagine that by changing the input parameters, we can fit our model to the observations in order to have an optimal representation of the true BL Lac blazar population with this parametrisation.

