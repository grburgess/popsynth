import numpy as np
from popsynth.spherical_population import ConstantSphericalPopulation
from popsynth.cosmological_population import SFRPopulation

from popsynth.pareto_population import ParetoPopulation
from popsynth.schechter_population import SchechterPopulation


class ParetoHomogeneousSphericalPopulation(ParetoPopulation, ConstantSphericalPopulation):

    def __init__(self, Lambda, Lmin, alpha, r_max=10, seed=1234):

        ParetoPopulation.__init__(
            self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='ParetoHomogeneousSphericalPopulation')
        ConstantSphericalPopulation.__init__(
            self, Lambda=Lambda, r_max=r_max, seed=seed, name='ParetoHomogeneousSphericalPopulation')


class SchechterHomogeneousSphericalPopulation(SchechterPopulation, ConstantSphericalPopulation):

    def __init__(self, Lambda, Lmin, alpha, r_max=10, seed=1234):

        SchechterPopulation.__init__(
            self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='SchechterHomogeneousSphericalPopulation')
        ConstantSphericalPopulation.__init__(
            self, Lambda=Lambda, r_max=r_max, seed=seed, name='SchechterHomogeneousSphericalPopulation')


class ParetoSFRPopulation(ParetoPopulation, SFRPopulation):

    def __init__(self, r0, rise, decay, peak, Lmin, alpha, r_max=10, seed=1234):

        ParetoPopulation.__init__(self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='ParetoSFRPopulation')
        SFRPopulation.__init__(
            self, r0=r0, rise=rise, decay=decay, peak=peak, r_max=r_max, seed=seed, name='ParetoSFRPopulation')


class SchechterSFRPopulation(SchechterPopulation, SFRPopulation):

    def __init__(self, r0, rise, decay, peak, Lmin, alpha, r_max=10, seed=1234):

        SchechterPopulation.__init__(
            self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='SchechterSFRPopulation')
        SFRPopulation.__init__(
            self, r0=r0, rise=rise, decay=decay, peak=peak, r_max=r_max, seed=seed, name='SchechterSFRPopulation')
