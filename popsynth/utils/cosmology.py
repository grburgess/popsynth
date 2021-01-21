from dataclasses import dataclass

import numpy as np
from astropy.constants import c
from astropy.cosmology import WMAP9 as cosmo
from numba import bool_, float32, float64
from numba.experimental import jitclass

_sol = c.value  # speed of light

spec = [
    #    ("_is_setup", bool_),
    ('Om', float32),
    ('h0', float32),
    # ('_dh', float64),
    # ('_om_reduced', float64),
    # ('_om_sqrt', float64),
    # ('_ode', float64),
]

Onu0 = cosmo.Onu0
Ogamma0 = cosmo.Ogamma0


@jitclass(spec)
class Cosmology(object):
    def __init__(self, Om=0.307, h0=67.7):

        #      self._is_setup = False

        self.Om = Om
        self.h0 = h0

        # self._dh =

        # self._Om_reduced =
        # self._Om_sqrt = np.sqrt(self.Om)
        # self._Ode =

        # self._is_setup = True

    def _setup(self):

        self._dh = _sol * 1.0e-3 / self.h0

        self._Om_reduced = (1 - self.Om) / self.Om
        self._Om_sqrt = np.sqrt(self.Om)
        self._Ode = 1 - self.Om - (cosmo.Onu0 + cosmo.Ogamma0)

    @property
    def Om_reduced(self):
        return (1 - self.Om) / self.Om

    @property
    def Om_sqrt(self):
        return np.sqrt(self.Om)

    @property
    def Ode(self):
        return 1 - self.Om - (Onu0 + Ogamma0)

    @property
    def dh(self):
        return _sol * 1.0e-3 / self.h0

    def Phi(self, x):
        x2 = x * x
        x3 = x * x * x
        top = 1.0 + 1.320 * x + 0.441 * x2 + 0.02656 * x3
        bottom = 1.0 + 1.392 * x + 0.5121 * x2 + 0.03944 * x3
        return top / bottom

    def xx(self, z):
        return self.Om_reduced / np.power(1.0 + z, 3)

    def luminosity_distance(self, z):
        """
        dL in cm
        """
        x = self.xx(z)
        z1 = 1.0 + z
        val = ((2 * self.dh * z1 / self.Om_sqrt) *
               (self.Phi(self.xx(0)) - 1.0 /
                (np.sqrt(z1)) * self.Phi(x)) * 3.086e24)  # in cm
        return val

    def a(self, z):

        return np.sqrt(np.power(1 + z, 3.0) * self.Om + self.Ode)

    def comoving_transverse_distance(self, z):

        return self.luminosity_distance(z) / (1.0 + z)

    def differential_comoving_volume(self, z):
        """
        differential comoving volume in Gpc2
        """

        td = self.comoving_transverse_distance(z) / 3.086e24

        return (self.dh * td * td / self.a(z)) * 1e-9  # Gpc^3

    # def __setattr__(self, name, value):

    #     super().__setattr__(name, value)

    #     print("oh shit")
    #     if (name in ["Om", "h0"]) and self._is_setup:

    #         self._setup()


cosmology = Cosmology()
