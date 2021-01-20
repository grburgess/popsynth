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

# Fun with the Milky Way

While not entirely useful at the moment. There is support for generating simplistic spiral galaxy distribtuions.

```python
import popsynth
from popsynth.populations.spatial_populations import MWRadialPopulation
```

```python
ld = popsynth.distributions.pareto_distribution.ParetoDistribution()
ld.alpha =3
ld.Lmin = 1
```

```python
synth = MWRadialPopulation(rho=1, luminosity_distribution=ld)
```

```python
population = synth.draw_survey(boundary=1E-2,  no_selection=True)
```

```python
population.display_obs_fluxes_sphere(cmap='magma', background_color='black', size=.1);
```

```python

```
