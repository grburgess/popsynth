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
`popsynth` core function is to create **observed** surveys from **latent** population models. 

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

`popsynth` allows you to specify these spatial and property
distributions in an object-oriented way to create surveys. The final
ingredient to creating a sample for a survey is knowing how many
objects to sample from the population (before any selection effects
are applied). Often, we see this number in simulation frameworks
presented as "we draw N objects to guarantee we have enough." This is
incorrect. A survey takes place over a given period of time ($\Delta
t$) in which observed objects are counted. This is a description of a
Poisson process. Thus, the number of objects in a simulation of this
survey is a draw from a Poisson distribution:

$$N \sim \mathrm{Poisson}\left(\Delta t \frac{d\Lambda}{dt}\right) \mathrm{.}$$

Thus, `popsynth` first numerically integrates the spatial
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
