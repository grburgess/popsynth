// variable Poisson intensity
vector dNdV(vector z, real r0, real rise, real decay, real peak) {

  int N = num_elements(z);

  vector[N] bottom;
  vector[N] top;
  vector[N] frac;

  top = r0 * (1. + rise*z);
  frac = z/peak;
  for (n in 1:N) {

    bottom[n] = 1+frac[n]^decay;

  }

  return top ./ bottom;
}

// Integrand of the rate
real[] N_integrand(real z, real[] state, real[] params, real[] x_r, int[] x_i) {

  real r0;
  real rise;
  real decay;
  real peak;

  real Om;
  real Ode;
  real Om_reduced;
  real Om_sqrt;
  real hubble_distance;
  real phi_0;


  real dstatedz[1];

  real tmp_dv;
  real tmp_z[1] = {z};

  r0 = params[1];
  rise = params[2];
  decay = params[3];
  peak = params[4];

  Om = params[5];
  Ode = params[6];
  hubble_distance = params[7];
  Om_reduced = params[8];
  Om_sqrt = params[9];
  phi_0 = params[10];

  tmp_dv = differential_comoving_volume(to_vector(tmp_z), Om, Ode, hubble_distance, Om_reduced,Om_sqrt,phi_0)[1] /(1+z);

  dstatedz[1] = dNdV_int(z, r0, rise, decay, peak) * tmp_dv;

  return dstatedz;
}
