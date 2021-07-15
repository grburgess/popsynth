---
title: 'popsynth: A generic astrophysical population synthesis framework'
tags:
  - Python
  - astronomy
  - population synthesis
  - cosmology
authors:
  - name: J. Michael Burgess
    orcid: 0000-0003-3345-9515
    affiliation: "1"
  - name: Francesca Capel
    orcid: 0000-0002-1153-2139
    affiliation: "2"
bibliography: paper.bib
affiliations:
 - name: Max Planck Institute for Extraterrestrial Physics, Giessenbachstrasse, 85748 Garching, Germany
   index: 1
 - name: Technical University of Munich, Boltzmannstrasse 2, 85748 Garching, Germany
   index: 2
date: "07 April 2021"
---

# Summary

Simulating a population survey of fluxes and redshifts (distances) from an
astrophysical population is a routine task. `popsynth` provides a
generic, object-oriented framework to produce synthetic surveys from
various distributions and luminosity functions, apply selection
functions the observed variables and store them in a portable (HDF5)
format. Population synthesis routines can be constructed either using
classes or from a serializable YAML format allowing flexibility and
portability. Users can not only sample the luminosity and distance of
the populations, but they can create auxiliary distributions for
parameters which can have arbitrarily complex dependencies on one
another. Thus, users can simulate complex astrophysical populations
which can be used to calibrate analysis frameworks or quickly test
ideas.

# Statement of need

`popsynth` provides a generic framework for simulating astrophysical
populations with an easily extensible class inheritance scheme that
allows users to adapt the code to their own needs. As understanding
the rate functions of astrophysical populations (e.g., gravitational
wave sources, gamma-ray bursts, active galactic nuclei) becomes an
increasingly important field [@Loredo:2019], researchers develop
various ways to estimate these populations from real data. `popsynth`
provides a way to calibrate these analysis techniques by producing
synthetic data where the inputs are known
[e.g. @Mortlock:2019]. Moreover, selection effects are an important
part of population analysis and the ability to include this property
when generating a population is vital to the calibration of any survey
analysis method which operates on an incomplete sample.

Similar frameworks exist for simulating data from specific catalogs
such as `SkyPy` [@skypy] and `firesong` [@firesong], however, these
have much more focused applications and do not include the ability to
impose selection functions.

# Procedure

Once a rate function and all associated distributions are specified in
`popsynth`, a numeric integral over the rate function produces the
total rate of objects in the populations. A survey is created by
making a draw from a Poisson distribution with mean equal to the total
rate of objects multiplied by survey duration for the number of
objects in the survey. For each object, the properties such as
distance and luminosity are sampled from their associated
distributions. Selection functions are then applied to latent or
observed variables as specified by the user. Finally, all population
objects and variables are returned in an object that can be serialized
to disk for later examination. Further details on the mathematics,
procedure, and details on customization can be found in the extensive
[documentation](https://popsynth.readthedocs.io/).


# Acknowledgments

This project was inspired by conversations with Daniel J. Mortlock
wherein we tried to calibrate an analysis method we will eventually
get around to finishing. Inspiration also came from wanting to
generalize the examples from Will Farr's lecture note
[@selection]. J. Michael Burgess acknowledges support from the
Alexander von Humboldt Stiftung. Francesca Capel acknowledges
financial support from the Excellence Cluster ORIGINS, which is funded
by the Deutsche Forschungsgemeinschaft (DFG, German Research
Foundation) under Germany’s Excellence Strategy
- EXC-2094-390783311.

# References
