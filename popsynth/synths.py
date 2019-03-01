import numpy as np
import pandas as pd
from popsynth.populations.spherical_population import ConstantSphericalPopulation, ZPowerSphericalPopulation
from popsynth.populations.cosmological_population import SFRPopulation, MergerPopulation, MadauPopulation

from popsynth.populations.pareto_population import ParetoPopulation
from popsynth.populations.schechter_population import SchechterPopulation
from popsynth.populations.log10_normal_population import Log10NormalPopulation
from popsynth.populations.log_normal_population import LogNormalPopulation
from popsynth.populations.bpl_population import BPLPopulation

class ParetoHomogeneousSphericalPopulation(ParetoPopulation, ConstantSphericalPopulation):

    def __init__(self, Lambda, Lmin, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda: 
        :param Lmin: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """
        

        ParetoPopulation.__init__(
            self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='ParetoHomogeneousSphericalPopulation')
        ConstantSphericalPopulation.__init__(
            self, Lambda=Lambda, r_max=r_max, seed=seed, name='ParetoHomogeneousSphericalPopulation')


class Log10NormalHomogeneousSphericalPopulation(Log10NormalPopulation, ConstantSphericalPopulation):

    def __init__(self, Lambda, mu, tau, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda: 
        :param mu: 
        :param tau: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        Log10NormalPopulation.__init__(
            self, mu=mu, tau=tau, r_max=r_max, seed=seed, name='Log10NormalHomogeneousSphericalPopulation')
        ConstantSphericalPopulation.__init__(
            self, Lambda=Lambda, r_max=r_max, seed=seed, name= 'Log10NormalHomogeneousSphericalPopulation')


class Log10NormalZPowerSphericalPopulation(Log10NormalPopulation, ZPowerSphericalPopulation):

    def __init__(self, Lambda, delta, mu, tau, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda: 
        :param delta: 
        :param mu: 
        :param tau: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        
        Log10NormalPopulation.__init__(
            self, mu=mu, tau=tau, r_max=r_max, seed=seed, name='Log10NormalZPowerSphericalPopulation')
        ZPowerSphericalPopulation.__init__(
            self, Lambda=Lambda, delta=delta, r_max=r_max, seed=seed, name='Log10NormalZPowerSphericalPopulation')

        

class SchechterHomogeneousSphericalPopulation(SchechterPopulation, ConstantSphericalPopulation):

    def __init__(self, Lambda, Lmin, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda: 
        :param Lmin: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """
        SchechterPopulation.__init__(
            self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='SchechterHomogeneousSphericalPopulation')
        ConstantSphericalPopulation.__init__(
            self, Lambda=Lambda, r_max=r_max, seed=seed, name='SchechterHomogeneousSphericalPopulation')


class ParetoSFRPopulation(ParetoPopulation, SFRPopulation):

    def __init__(self, r0, rise, decay, peak, Lmin, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param r0: 
        :param rise: 
        :param decay: 
        :param peak: 
        :param Lmin: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        ParetoPopulation.__init__(self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='ParetoSFRPopulation')
        SFRPopulation.__init__(
            self, r0=r0, rise=rise, decay=decay, peak=peak, r_max=r_max, seed=seed, name='ParetoSFRPopulation')


    def generate_stan_code(self, stan_gen, **kwargs):

        ParetoPopulation.generate_stan_code(self, stan_gen, **kwargs)
        SFRPopulation.generate_stan_code(self, stan_gen, **kwargs)
        
        
class BPLSFRPopulation(BPLPopulation, SFRPopulation):

    def __init__(self, r0, rise, decay, peak, Lmin, alpha, Lbreak, beta, Lmax,  r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param r0: 
        :param rise: 
        :param decay: 
        :param peak: 
        :param Lmin: 
        :param alpha: 
        :param Lbreak: 
        :param beta: 
        :param Lmax: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        BPLPopulation.__init__(self, Lmin=Lmin, alpha=alpha, Lbreak=Lbreak, beta=beta, Lmax=Lmax,  r_max=r_max, seed=seed, name='BPLSFRPopulation')
        SFRPopulation.__init__(
            self, r0=r0, rise=rise, decay=decay, peak=peak, r_max=r_max, seed=seed, name='BPLSFRPopulation')


    def generate_stan_code(self, stan_gen, **kwargs):
        pass
#        ParetoPopulation.generate_stan_code(self, stan_gen, **kwargs)
 #       SFRPopulation.generate_stan_code(self, stan_gen, **kwargs)
        

        
class SchechterSFRPopulation(SchechterPopulation, SFRPopulation):

    def __init__(self, r0, rise, decay, peak, Lmin, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param r0: 
        :param rise: 
        :param decay: 
        :param peak: 
        :param Lmin: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        SchechterPopulation.__init__(
            self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='SchechterSFRPopulation')
        SFRPopulation.__init__(
            self, r0=r0, rise=rise, decay=decay, peak=peak, r_max=r_max, seed=seed, name='SchechterSFRPopulation')


class LogNormalSFRPopulation(LogNormalPopulation, SFRPopulation):

    def __init__(self, r0, rise, decay, peak, mu, tau, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param r0: 
        :param rise: 
        :param decay: 
        :param peak: 
        :param mu: 
        :param tau: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        LogNormalPopulation.__init__(
            self, mu=mu, tau=tau, r_max=r_max, seed=seed, name='LogNormalSFRPopulation')
        SFRPopulation.__init__(
            self, r0=r0, rise=rise, decay=decay, peak=peak, r_max=r_max, seed=seed, name='LogNormalSFRPopulation')




class ParetoMergerPopulation(ParetoPopulation, MergerPopulation):

    def __init__(self, r0, td, sigma, Lmin, alpha, r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param r0: 
        :param rise: 
        :param decay: 
        :param peak: 
        :param Lmin: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """

        ParetoPopulation.__init__(self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='ParetoMergerPopulation')
        MergerPopulation.__init__(
            self, r0=r0, td=td, sigma=sigma,  r_max=r_max, seed=seed, name='ParetoMergerPopulation')


    def generate_stan_code(self, stan_gen, **kwargs):

        pass
        # ParetoPopulation.generate_stan_code(self, stan_gen, **kwargs)
        # SFRPopulation.generate_stan_code(self, stan_gen, **kwargs)


class ParetoMaduaPopulation(ParetoPopulation, MadauPopulation):

    def __init__(self, r0, Lmin, alpha,  r_max=10, seed=1234):
        """FIXME! briefly describe function

        :param Lambda: 
        :param Lmin: 
        :param alpha: 
        :param r_max: 
        :param seed: 
        :returns: 
        :rtype: 

        """
        

        ParetoPopulation.__init__(
            self, Lmin=Lmin, alpha=alpha, r_max=r_max, seed=seed, name='ParetoMaduaPopulation')
        MadauPopulation.__init__(
            self, r0=r0, r_max=r_max, seed=seed, name='ParetoMaduaPopulation')
