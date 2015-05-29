import mpmath as mp
import numpy as np
import scipy.integrate as scint
from .util import (zp, zpd, j0, f1, f2)
from .util import (boltzmann, emass, echarge, permittivity, cspeed)

from scipy.special import j1, itj0y0

class Noise(object):
    def __init__(self, ant_len, ant_rad, base_cap):
        self.ant_len = ant_len
        self.ant_rad = ant_rad
        self.al_ratio = ant_rad/ant_len
        self.base_cap = base_cap
        self.z_unit = 8.313797e6
        self.v_unit = 1.62760e-15

    @staticmethod
    def fperp(x):
        """
        result of angular integration in proton noise integration.

        """
        return 8./x * (2*itj0y0(x)[0] - itj0y0(2*x)[0] + j1(2*x) - 2*j1(x))

    def proton(self, wc, l, tep, tc, vsw):
        """
        proton noise.
        wc: w/w_c, where w_c is the core electron plasma frequency.
        l: l_ant/l_dc, where l_ant is antenna length
        l_dc: core electron debye length.
        tep: T_e/T_p
        tc: core electron temperature
        vsw: solar wind speed.

        """
        vtc = np.sqrt(2 * boltzmann * tc/ emass)
        omega = wc * vtc/vsw /np.sqrt(2.)
        integrand = lambda y: y * fperp(y * l)/ (y**2 + 1 + omega**2) / (y**2 + 1 + omega**2 + tep)
        return scint.quad(integrand, 0, inf, epsrel = 1.e-5) * boltzmann * tc/ (2 * np.pi * permittivity * vsw)

        
