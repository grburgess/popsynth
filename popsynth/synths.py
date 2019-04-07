import numpy as np
import pandas as pd

from popsynth.populations.spatial_populations import SphericalPopulation, ZPowerSphericalPopulation, SFRPopulation

from popsynth.distributions.log10_normal_distribution import Log10NormalDistribution
from popsynth.distributions.log_normal_distribution import LogNormalDistribution
from popsynth.distributions.bpl_distribution import BPLPopulation

from popsynth.population_synth import PopulationSynth




        

####

        
class Log10NormalHomogeneousSphericalPopulation(
    Log10NormalDistribution, ConstantSphericalDistribution
):
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

        Log10NormalDistribution.__init__(
            self,
            mu=mu,
            tau=tau,
            r_max=r_max,
            seed=seed,
            name="Log10NormalHomogeneousSphericalPopulation",
        )
        ConstantSphericalDistribution.__init__(
            self,
            Lambda=Lambda,
            r_max=r_max,
            seed=seed,
            name="Log10NormalHomogeneousSphericalPopulation",
        )


class Log10NormalZPowerSphericalPopulation(
    Log10NormalDistribution, ZPowerSphericalDistribution
):
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

        Log10NormalDistribution.__init__(
            self,
            mu=mu,
            tau=tau,
            r_max=r_max,
            seed=seed,
            name="Log10NormalZPowerSphericalPopulation",
        )
        ZPowerSphericalDistribution.__init__(
            self,
            Lambda=Lambda,
            delta=delta,
            r_max=r_max,
            seed=seed,
            name="Log10NormalZPowerSphericalPopulation",
        )




# class ParetoSFRPopulation(ParetoDistribution, SFRDistribtution):
#     def __init__(self, r0, rise, decay, peak, Lmin, alpha, r_max=10, seed=1234):
#         """FIXME! briefly describe function

#         :param r0: 
#         :param rise: 
#         :param decay: 
#         :param peak: 
#         :param Lmin: 
#         :param alpha: 
#         :param r_max: 
#         :param seed: 
#         :returns: 
#         :rtype: 

#         """

#         ParetoDistribution.__init__(
#             self,
#             Lmin=Lmin,
#             alpha=alpha,
#             r_max=r_max,
#             seed=seed,
#             name="ParetoSFRPopulation",
#         )
#         SFRDistribtution.__init__(
#             self,
#             r0=r0,
#             rise=rise,
#             decay=decay,
#             peak=peak,
#             r_max=r_max,
#             seed=seed,
#             name="ParetoSFRPopulation",
#         )

#     def generate_stan_code(self, stan_gen, **kwargs):

#         ParetoDistribution.generate_stan_code(self, stan_gen, **kwargs)
#         SFRDistribtution.generate_stan_code(self, stan_gen, **kwargs)


class BPLSFRPopulation(BPLPopulation, SFRDistribtution):
    def __init__(
        self,
        r0,
        rise,
        decay,
        peak,
        Lmin,
        alpha,
        Lbreak,
        beta,
        Lmax,
        r_max=10,
        seed=1234,
    ):
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

        BPLPopulation.__init__(
            self,
            Lmin=Lmin,
            alpha=alpha,
            Lbreak=Lbreak,
            beta=beta,
            Lmax=Lmax,
            r_max=r_max,
            seed=seed,
            name="BPLSFRPopulation",
        )
        SFRDistribtution.__init__(
            self,
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            name="BPLSFRPopulation",
        )

    def generate_stan_code(self, stan_gen, **kwargs):
        pass


#        ParetoDistribution.generate_stan_code(self, stan_gen, **kwargs)
#       SFRDistribtution.generate_stan_code(self, stan_gen, **kwargs)




class LogNormalSFRPopulation(LogNormalDistribution, SFRDistribtution):
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

        LogNormalDistribution.__init__(
            self, mu=mu, tau=tau, r_max=r_max, seed=seed, name="LogNormalSFRPopulation"
        )
        SFRDistribtution.__init__(
            self,
            r0=r0,
            rise=rise,
            decay=decay,
            peak=peak,
            r_max=r_max,
            seed=seed,
            name="LogNormalSFRPopulation",
        )


class ParetoMergerPopulation(ParetoDistribution, MergerDistribution):
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

        ParetoDistribution.__init__(
            self,
            Lmin=Lmin,
            alpha=alpha,
            r_max=r_max,
            seed=seed,
            name="ParetoMergerPopulation",
        )
        MergerDistribution.__init__(
            self,
            r0=r0,
            td=td,
            sigma=sigma,
            r_max=r_max,
            seed=seed,
            name="ParetoMergerPopulation",
        )

    def generate_stan_code(self, stan_gen, **kwargs):

        pass
        # ParetoDistribution.generate_stan_code(self, stan_gen, **kwargs)
        # SFRDistribtution.generate_stan_code(self, stan_gen, **kwargs)
