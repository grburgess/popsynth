import numpy as np
from popsynth.spherical_population import ConstantSphericalPopulation, ZPowerSphericalPopulation
from popsynth.cosmological_population import SFRPopulation

from popsynth.pareto_population import ParetoPopulation
from popsynth.schechter_population import SchechterPopulation
from popsynth.log10_normal_population import Log10NormalPopulation
from popsynth.log_normal_population import LogNormalPopulation
from popsynth.bpl_population import BPLPopulation

class ParetoHomogeneousSphericalPopulation(ParetoPopulation, ConstantSphericalPopulation):

    def __init__(self, Lambda, Lmin, alpha, r_max=10, seed=1234):

        ParetoPopulation.__init__(
            self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='ParetoHomogeneousSphericalPopulation')
        ConstantSphericalPopulation.__init__(
            self, Lambda=Lambda, r_max=r_max, seed=seed, name='ParetoHomogeneousSphericalPopulation')


class Log10NormalHomogeneousSphericalPopulation(Log10NormalPopulation, ConstantSphericalPopulation):

    def __init__(self, Lambda, mu, tau, r_max=10, seed=1234):

        Log10NormalPopulation.__init__(
            self, mu=mu, tau=tau, r_max=r_max, seed=seed, name='Log10NormalHomogeneousSphericalPopulation')
        ConstantSphericalPopulation.__init__(
            self, Lambda=Lambda, r_max=r_max, seed=seed, name= 'Log10NormalHomogeneousSphericalPopulation')


class Log10NormalZPowerSphericalPopulation(Log10NormalPopulation, ZPowerSphericalPopulation):

    def __init__(self, Lambda, delta, mu, tau, r_max=10, seed=1234):

        Log10NormalPopulation.__init__(
            self, mu=mu, tau=tau, r_max=r_max, seed=seed, name='Log10NormalZPowerSphericalPopulation')
        ZPowerSphericalPopulation.__init__(
            self, Lambda=Lambda, delta=delta, r_max=r_max, seed=seed, name='Log10NormalZPowerSphericalPopulation')

        

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


    def generate_stan_code(self, stan_gen, **kwargs):

        ParetoPopulation.generate_stan_code(self, stan_gen, **kwargs)
        SFRPopulation.generate_stan_code(self, stan_gen, **kwargs)
        

class BPLSFRPopulation(BPLPopulation, SFRPopulation):

    def __init__(self, r0, rise, decay, peak, Lmin, alpha, Lbreak, beta, Lmax,  r_max=10, seed=1234):

        BPLPopulation.__init__(self, Lmin=Lmin, alpha=alpha, Lbreak=Lbreak, beta=beta, Lmax=Lmax,  r_max=r_max, seed=seed, name='BPLSFRPopulation')
        SFRPopulation.__init__(
            self, r0=r0, rise=rise, decay=decay, peak=peak, r_max=r_max, seed=seed, name='BPLSFRPopulation')


    def generate_stan_code(self, stan_gen, **kwargs):
        pass
#        ParetoPopulation.generate_stan_code(self, stan_gen, **kwargs)
 #       SFRPopulation.generate_stan_code(self, stan_gen, **kwargs)
        

        
class SchechterSFRPopulation(SchechterPopulation, SFRPopulation):

    def __init__(self, r0, rise, decay, peak, Lmin, alpha, r_max=10, seed=1234):

        SchechterPopulation.__init__(
            self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='SchechterSFRPopulation')
        SFRPopulation.__init__(
            self, r0=r0, rise=rise, decay=decay, peak=peak, r_max=r_max, seed=seed, name='SchechterSFRPopulation')


class LogNormalSFRPopulation(LogNormalPopulation, SFRPopulation):

    def __init__(self, r0, rise, decay, peak, mu, tau, r_max=10, seed=1234):

        LogNormalPopulation.__init__(
            self, mu=mu, tau=tau, r_max=r_max, seed=seed, name='LogNormalSFRPopulation')
        SFRPopulation.__init__(
            self, r0=r0, rise=rise, decay=decay, peak=peak, r_max=r_max, seed=seed, name='LogNormalSFRPopulation')
