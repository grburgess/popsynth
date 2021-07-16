
![CI](https://github.com/grburgess/popsynth/workflows/CI/badge.svg?branch=master)
[![codecov](https://codecov.io/gh/grburgess/popsynth/branch/master/graph/badge.svg)](https://codecov.io/gh/grburgess/popsynth)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/5d02c9e6f5c540989a615eb1575863e3)](https://app.codacy.com/gh/grburgess/popsynth?utm_source=github.com&utm_medium=referral&utm_content=grburgess/popsynth&utm_campaign=Badge_Grade_Settings)
[![Documentation Status](https://readthedocs.org/projects/popsynth/badge/?version=latest)](https://popsynth.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5109590.svg)](https://doi.org/10.5281/zenodo.5109590)
![PyPI](https://img.shields.io/pypi/v/popsynth)
![PyPI - Downloads](https://img.shields.io/pypi/dm/popsynth)
 [![status](https://joss.theoj.org/papers/a52e4c2c355396e7946917996502aac0/status.svg)](https://joss.theoj.org/papers/a52e4c2c355396e7946917996502aac0)
# popsynth

![alt text](https://raw.githubusercontent.com/grburgess/popsynth/master/external/logo.png)

`popsynth` core function is to create **observed** surveys from **latent** population models. 

First, let's define what a population of objects is in terms of a
generative model. The two main ingredients are the objects' spatial
distribution (<img src="https://render.githubusercontent.com/render/math?math=\lambda(\vec{r},\vec{\psi})">) and the distribution of
their inherent properties (<img src="https://render.githubusercontent.com/render/math?math=\pi(\vec{\phi} | \vec{\psi})">). Here,
<img src="https://render.githubusercontent.com/render/math?math=\vec{\psi}"> are the latent population parameters, <img src="https://render.githubusercontent.com/render/math?math=\vec{r}"> are the
spatial locations of the objects, and <img src="https://render.githubusercontent.com/render/math?math=\vec{\phi}"> are the properties
of the individual objects (luminosity, spin, viewing angle, mass,
etc.). The spatial distribution is defined such that:

<img src="https://render.githubusercontent.com/render/math?math=\frac{d \Lambda}{dt}(\vec{\psi}) = \int d r \frac{dV}{dr} \lambda(\vec{r}, \vec{\psi}))">

is the intensity of objects for a given set of population
parameters. With these definitions we can define the probability for
an object to have position <img src="https://render.githubusercontent.com/render/math?math=\vec{r}"> and properties <img src="https://render.githubusercontent.com/render/math?math=\vec{\phi}"> as

<img src="https://render.githubusercontent.com/render/math?math=\pi(\vec{r}, \vec{\phi} | \vec{\psi}) = \frac{\lambda(\vec{r}, \vec{\psi})  \pi(\vec{\phi} | \vec{\psi})}{ \int d r \frac{dV}{dr} \lambda(\vec{r}, \vec{\psi})}">

`popsynth` allows you to specify these spatial and property
distributions in an object-oriented way to create surveys. The final
ingredient to creating a sample for a survey is knowing how many
objects to sample from the population (before any selection effects
are applied). Often, we see this number in simulation frameworks
presented as "we draw N objects to guarantee we have enough." This is
incorrect. A survey takes place over a given period of time (<img src="https://render.githubusercontent.com/render/math?math=\Delta t">) in which observed objects are counted. This is a description of a
Poisson process. Thus, the number of objects in a simulation of this
survey is a draw from a Poisson distribution:

<img src="https://render.githubusercontent.com/render/math?math=N \sim Poisson \left( \Delta t \frac{d\Lambda}{dt} \right)">

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


## Installation
```bash
pip install popsynth
```


Note: **This is not synth pop!** If you were looking for some hard driving beats out of a yamaha keyboard with bells... look elsewhere

![alt text](https://raw.githubusercontent.com/grburgess/popsynth/master/external/pop.gif)


## Contributing

Contributions to ```popsynth``` are always welcome. They can come in the form of:

### Bug reports

Please use the [Github issue tracking system for any
bugs](https://github.com/grburgess/popsynth/issues), for questions,
and or feature requests.

### Code and more distributions

While it is easy to create custom distributions in your local setup,
if you would like to add them to popsynth directly, go ahead. Please
include tests to ensure that your contributions are compatible with
the code and can be maintained in the long term.

### Documentation

Additions or examples, tutorials, or better explanations are always
welcome. To ensure that the documentation builds with the current
version of the software, I am using
[jupytext](https://jupytext.readthedocs.io/en/latest/) to write the
documentation in Markdown. These are automatically converted to and
executed as jupyter notebooks when changes are pushed to Github.


