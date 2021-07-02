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

# Fun with the Milky Way

While not entirely useful at the moment. There is support for generating simplistic spiral galaxy distribtuions.

```python
import popsynth
import ipyvolume as ipv


from astropy.coordinates import SkyCoord

%matplotlib inline

import matplotlib.pyplot as plt
from jupyterthemes import jtplot

purple = "#B833FF"

popsynth.update_logging_level("INFO")
from popsynth.populations.spatial_populations import MWRadialPopulation
```

```python
ld = popsynth.distributions.pareto_distribution.ParetoDistribution()
ld.alpha = 3
ld.Lmin = 1
```

```python
synth = MWRadialPopulation(rho=1, luminosity_distribution=ld)
```

```python
population = synth.draw_survey()
```

```python
fig = population.display_obs_fluxes_sphere(
    cmap="magma", background_color="black", size=0.1
)
```

