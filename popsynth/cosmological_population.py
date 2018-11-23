import numpy as np


from astropy.cosmology import WMAP9 as cosmo

from popsynth.population_synth import PopulationSynth
from popsynth.utils.package_data import copy_package_data

from astropy.constants import c as sol
sol = sol.value


h0 = 69.3
dh = sol * 1.e-3 / h0
Om = 0.286
Om_reduced = ((1-Om)/Om)
Om_sqrt = np.sqrt(Om)
Ode = 1 - Om -(cosmo.Onu0 + cosmo.Ogamma0)


def Phi(x):    
    x2 = x*x
    x3 = x*x*x 
    top = 1. + 1.320*x + 0.441 *x2 + 0.02656*x3
    bottom = 1. + 1.392*x + 0.5121*x2 + 0.03944*x3
    return top/bottom

def xx(z): 
    return Om_reduced / np.power(1+z,3)


def luminosity_distance(z):
    x = xx(z)
    z1 = 1+z
    val = (2 * dh * z1 / Om_sqrt) *( Phi(xx(0)) - 1./(np.sqrt(z1)) * Phi(x)) *3.086E24 # in cm
    return val

def a(z):
    return np.sqrt( np.power(1+z,3.)*Om +Ode)

def comoving_transverse_distance(z):
    return luminosity_distance(z)/(1.+z)

def differential_comoving_volume(z):
    td = (comoving_transverse_distance(z)/3.086E24) 
    return (dh*td*td/a(z)) * 1E-9 # Gpc^3


class CosmologicalPopulation(PopulationSynth):

    def __init__(self, r_max=10, seed=1234, name='cosmo'):

        PopulationSynth.__init__(self, r_max, seed, name)

    def differential_volume(self, z):

        td = (comoving_transverse_distance(z)/3.086E24) 
        return (dh*td*td/a(z)) * 1E-9 # Gpc^3


    def transform(self, L, z):

        return L/(4. * np.pi * luminosity_distance(z)**2)

    def time_adjustment(self, z):
        return (1+z)


    def generate_stan_code(self, stan_gen, **kwargs):

        # add the cosmology code
        copy_package_data('cosmo.stan')
        copy_package_data('cosmo_constants.stan')

        stan_gen.blocks['functions'].add_include('cosmo.stan')
        stan_gen.blocks['transformed_data'].add_include('cosmo_constants.stan')

        if 'distance_flag' in kwargs:
            distance_flag = kwargs['distance_flag']
        else:

            distance_flag = False
            
        if not distance_flag:
            code = """
            // technical varaibales required for the integration
            real x_r[0];
            int x_i[0];

            real zout[1];
            real state0[1];

            // here we precompute things which depend on the data only

            vector[Nz] z2_inverse;
            vector[N] log_p_obs;
            vector[Nz] log_volume_element;
            real total_static_prob;


            zout[1] = z_max;
            state0[1] = 0.0;

            // the detection probability and comoving volume elements
            // for the observed objects is static
            // we precompute and then sum
            log_volume_element = log(differential_comoving_volume(known_z_obs, Om, Ode, hubble_distance, Om_reduced,Om_sqrt,phi_0)) - log(1+z_obs);
            log_p_obs = log_p_det(log_flux_obs, boundary, strength);
            total_static_prob = sum(log_p_obs) + sum(log_volume_element);

            // the transformation of the objects is static
            z2_inverse = transform(to_vector(rep_array(1.,Nn)), known_z_obs , hubble_distance, Om_reduced, Om_sqrt, phi_0);


            """


        else:
            code = """
            // technical varaibales required for the integration
            real x_r[0];
            int x_i[0];

            real zout[1];
            real state0[1];

            // here we precompute things which depend on the data only

            vector[N] z2_inverse;
            vector[N] log_p_obs;
            vector[N] log_volume_element;
            real total_static_prob;


            zout[1] = z_max;
            state0[1] = 0.0;

            // the detection probability and comoving volume elements
            // for the observed objects is static
            // we precompute and then sum
            log_volume_element = log(differential_comoving_volume(z_obs, Om, Ode, hubble_distance, Om_reduced,Om_sqrt,phi_0)) - log(1+z_obs);
            log_p_obs = log_p_det(log_flux_obs, boundary, strength);
            total_static_prob = sum(log_p_obs) + sum(log_volume_element);

            // the transformation of the objects is static
            z2_inverse = transform(to_vector(rep_array(1.,N)), z_obs , hubble_distance, Om_reduced, Om_sqrt, phi_0);


            """

        stan_gen.blocks['transformed_data'].insert_code(code)



        code = """

         // setup for the integral
        real Lambda; // this is total Lambda!
        real params[10];
        real integration_result[1,1];


        vector[M] log_flux_tilde_latent;
        vector[M] log_dndv_tilde;
        vector[M] log_dvdz_tilde;
        vector[M] log_pndet;

        

        """


        stan_gen.blocks['model'].insert_code(code)


        if not distance_flag:

            code = """
        
            log_flux_obs ~ normal(log10(flux_latent), flux_sigma);
            
            // add the differential of the inhomogeneous process on
            // and detection probability
        
            target += total_static_prob;
            target += sum(log(dNdV(known_z_obs, r0, rise, decay, peak)));
            
            
            // unkown z
            target += sum( log( differential_comoving_volume(z, Om, Ode, hubble_distance, Om_reduced,Om_sqrt,phi_0) ) - log(1+z) );
            target += sum( log( dNdV(z, r0, rise, decay, peak) ) );
            """


        else:

            code = """
        
            log_flux_obs ~ normal(log10(flux_latent), flux_sigma);


            // add the differential of the inhomogeneous process on
            // and detection probability

            target += total_static_prob;
            target += sum(log(dNdV(z_obs, r0, rise, decay, peak)));

            """


        stan_gen.blocks['model'].insert_code(code)

        code = """



        log_flux_tilde_latent = log10(transform(lum_tilde_latent, z_tilde, hubble_distance, Om_reduced, Om_sqrt, phi_0));
        

        log_dndv_tilde =  log(dNdV(z_tilde, r0, rise, decay, peak));
        log_dvdz_tilde = log(differential_comoving_volume(z_tilde, Om, Ode, hubble_distance, Om_reduced,Om_sqrt,phi_0))-log(1+z_tilde);
        log_pndet = log_p_ndet(log_flux_tilde, boundary, strength) ;

        for (m in 1:M) {


           target+= log_sum_exp( log_dndv_tilde[m]
                          + log_dvdz_tilde[m]
                          + log_pndet[m]
                          + normal_lpdf(log_flux_tilde[m] | log_flux_tilde_latent[m], flux_sigma)
                          //insert here
                          ,

                          log(Lambda0) + normal_lpdf(log_flux_tilde[m]| log10(boundary),4)+
                          // insert here
                          + uniform_lpdf(z_tilde[m]| 0 ,z_max)

                          );



        }

        // Poisson normalization for the integral rate
        
        // (Distinguishiable) Poisson process model

        params[1] = r0;
        params[2] = rise;
        params[3] = decay;
        params[4] = peak;
        params[5] = Om;
        params[6] = Ode;
        params[7] = hubble_distance;
        params[8] = Om_reduced;
        params[9] = Om_sqrt;
        params[10] = phi_0;


        // integrate the dN/dz to get the normalizing constant for given r0 and alpha
        integration_result = integrate_ode_rk45(N_integrand, state0, 0.0, zout, params, x_r, x_i);
        Lambda = integration_result[1,1];

        
        
        target += - Lambda - Lambda0;
        """

        stan_gen.blocks['model'].insert_code(code)
        
class SFRPopulation(CosmologicalPopulation):

    def __init__(self, r0, rise, decay, peak, r_max=10, seed=1234, name='_sfrcosmo'):

        CosmologicalPopulation.__init__(self, r_max, seed, name)

        self.set_spatial_distribution_params(r0=r0, rise=rise, decay=decay, peak=peak)

        self._spatial_form = r'\rho_0 \frac{1+r \cdot z}{1+ \left(z/p\right)^d}'


    def dNdV(self, z):
        top = 1. + self.rise * z
        bottom = 1+np.power(z/self.peak, self.decay)
    
        return self.r0 * top/bottom

        
    def __get_r0(self):
             """Calculates the 'r0' property."""
             return self._spatial_params['r0']

    def ___get_r0(self):
         """Indirect accessor for 'r0' property."""
         return self.__get_r0()

    def __set_r0(self, r0):
         """Sets the 'r0' property."""
         self.set_spatial_distribution_params(r0=r0, rise=self.rise, decay=self.decay, peak=self.peak)

    def ___set_r0(self, r0):
         """Indirect setter for 'r0' property."""
         self.__set_r0(r0)

    r0 = property(___get_r0, ___set_r0,
                     doc="""Gets or sets the r0.""")

    def __get_rise(self):
             """Calculates the 'rise' property."""
             return self._spatial_params['rise']

    def ___get_rise(self):
         """Indirect accessor for 'rise' property."""
         return self.__get_rise()

    def __set_rise(self, rise):
         """Sets the 'rise' property."""
         self.set_spatial_distribution_params(r0=self.r0, rise=rise, decay=self.decay, peak=self.peak)

    def ___set_rise(self, rise):
         """Indirect setter for 'rise' property."""
         self.__set_rise(rise)

    rise = property(___get_rise, ___set_rise,
                     doc="""Gets or sets the rise.""")

    
    def __get_decay(self):
             """Calculates the 'decay' property."""
             return self._spatial_params['decay']

    def ___get_decay(self):
         """Indirect accessor for 'decay' property."""
         return self.__get_decay()

    def __set_decay(self, decay):
         """Sets the 'decay' property."""
         self.set_spatial_distribution_params(r0=self.r0, rise=self.rise, decay=decay, peak=self.peak)

    def ___set_decay(self, decay):
         """Indirect setter for 'decay' property."""
         self.__set_decay(decay)

    decay = property(___get_decay, ___set_decay,
                     doc="""Gets or sets the decay.""")


    def __get_peak(self):
        """Calculates the 'peak' property."""
        return self._spatial_params['peak']

    def ___get_peak(self):
         """Indirect accessor for 'peak' property."""
         return self.__get_peak()

    def __set_peak(self, peak):
         """Sets the 'peak' property."""
         self.set_spatial_distribution_params(r0=self.r0, rise=self.rise, decay=self.decay, peak=peak)

    def ___set_peak(self, peak):
         """Indirect setter for 'peak' property."""
         self.__set_peak(peak)

    peak = property(___get_peak, ___set_peak,
                     doc="""Gets or sets the peak.""")


    def generate_stan_code(self, stan_gen, **kwargs):


        CosmologicalPopulation.generate_stan_code(self, stan_gen, **kwargs)

        copy_package_data('sfr_functions.stan')
        stan_gen.blocks['functions'].add_include('sfr_functions.stan')

        
