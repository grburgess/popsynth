real h0 = 69.3; // km / (Mpc s)
real c = 299792.458; // km/s
real hubble_distance = c / h0; // Mpc
real Om = 0.286;
real Om_reduced = ((1.-Om)/Om);
real Om_sqrt = sqrt(Om);
real Onu0 = 3.5503713827562275e-05;
real Ogamma0 = 5.142438100667033e-05;
real Ode = 1 - Om -(Onu0 + Ogamma0);

real tmp_zero[1] = {0.};
real phi_0 = phi(psi(to_vector(tmp_zero), Om_reduced))[1];
