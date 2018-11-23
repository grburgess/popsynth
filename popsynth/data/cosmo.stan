vector psi( vector z, real Om_reduced) {

  int nz = num_elements(z);
  vector[nz] zp1 = z+1.;
  vector[nz] zzz = zp1 .* zp1 .* zp1;

  return Om_reduced ./ zzz;


}

vector phi(vector x) {

  int nx =  num_elements(x);
  vector[nx] xx = x .* x;
  vector[nx] xxx = xx .* x;
  vector[nx] top;
  vector[nx] bottom;

  top = 1. + 1.320*x + 0.441*xx + 0.02656*xxx;
  bottom = 1. + 1.392*x + 0.5121*xx + 0.03944*xxx;

  return top ./ bottom;

}

vector luminosity_distance(vector z, real hubble_distance, real Om_reduced, real Om_sqrt, real phi_0) {
  int nz =  num_elements(z);
  vector[nz] zp1 = z+1.;
  vector[nz] x = psi(z, Om_reduced);

  return (2 * hubble_distance * zp1 / Om_sqrt) .* ( phi_0 - inv_sqrt(zp1) .* phi(x)) *3.086E24; // in cm


}

vector comoving_transverse_distance(vector z, real hubble_distance, real Om_reduced, real Om_sqrt, real phi_0) {

  return luminosity_distance(z, hubble_distance, Om_reduced, Om_sqrt, phi_0) ./ (1.+z);

}

vector differential_comoving_volume(vector z, real Om, real Ode, real hubble_distance, real Om_reduced, real Om_sqrt, real phi_0) {
  int nz = num_elements(z);
  vector[nz] trans_dist = comoving_transverse_distance(z, hubble_distance,  Om_reduced, Om_sqrt, phi_0)/3.086E24;
  vector[nz] zp1 = z+1.;
  vector[nz] zzz = zp1 .* zp1 .* zp1;
  vector[nz] az_inv = inv_sqrt(zzz*Om + Ode);

  return 1E-9 * hubble_distance * square(trans_dist) .* az_inv; // Gpc^3

}


// transform flux into luminosity

vector transform(vector x, vector z, real hubble_distance, real Om_reduced, real Om_sqrt, real phi_0) {

  return x .* inv_square(luminosity_distance(z, hubble_distance, Om_reduced, Om_sqrt, phi_0)) /(4* pi()) ;
}

