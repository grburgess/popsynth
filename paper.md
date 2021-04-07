---
title: 'popsynth: A generic astrophysical population synthesis code'
tags:
  - Python
  - astronomy
  - population synthesis
  - cosmology
authors:
  - name: J. Michael Burgess
    orcid: 0000-0003-0872-7098
    affiliation: "1, 2" # 
  
affiliations:
 - name: Max Planck Institute for Extraterrestrial Physics, Giessenbachstrasse, 85741 Garching, Germany
   index: 1
 - name: Alexander von Humboldt Research Fellow
   index: 2

date: 07 April 2021
bibliography: paper.bib


---

# Summary
Simulating a population of fluxes and redshifts (distances) from an
astrophysical population is a routine task. `popsynth` provides a
generic, object-oriented framework to produce synthetic surveys from
various distributions and luminosity functions, apply selection
functions the observed variables and store them in a portable (HDF5)
format. Users can not only sample the luminosity and distance of the
populations, but they can create auxiliary distributions for
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
increasingly important field, researchers develop various ways to
estimate these populations from real data. `popsynth` provides a way
to calibrate these analysis techniques by producing synthetic data
where the inputs are known. Moreover, selection effects are an
important part of population analysis and the ability to include this
property when generating a population is vital to the calibration of
any survey analysis method which operates on an incomplete sample.

Similar frameworks exist for simulating data from specific catalogs
such as SkyPy and firesong, however, these have a much more focused
application and do not include the ability to impose selection functions. 


# Procedure

Once a rate function and all assocaited distributions are specified in
`popsynth`, a numeric integral over the rate function produces the
total rate of objects in the populations.

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.


A survey is created by
making a draw from a Poisson distribution with mean equal to the total
rate of objects for the number of objects in the survey. For each
object, the properties such as distance and luminosity are sampled
from their associated distributions. Selection functions are then
applied to latent or observed variables as specified by the
user. Finally, all population objects and variables are returned in an
object that can be serialized to disk for later examination.

# Documentation



# Citations

Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

<!-- For a quick reference, the following citation commands can be used: -->
<!-- - `@author:2001`  ->  "Author et al. (2001)" -->
<!-- - `[@author:2001]` -> "(Author et al., 2001)" -->
<!-- - `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)" -->

# Acknowledgments
This project was inspired by conversations with Daniel J. Mortlock
wherein we tried to calibrate an analysis method we will eventually
get around to finishing. Inspiration also came from wanting to
generalize the examples from Will Farr's lecture note (link). I am
thankful to contributions and critique from Francesca Capel.

# References