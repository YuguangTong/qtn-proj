import mpmath as mp
import numpy as np
import scipy.integrate as scint
from .util import (zp, zpd, zp2d, j0, f1, f2, fperp, timing)
from .util import (boltzmann, emass, echarge, permittivity, cspeed)


# These class provides methods to calculate
#   electron noise
#   proton noise
#   shot noise
#   transfer function (gain)
# for a specified (long) dipole wire antenna
#
# The implementation uses expressions from the classical paper by N. Meyer-Vernet et al. (1989)
#         https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JA094iA03p02405
# The proton noise calculation follows Issautier et al. (1999)
#         https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/1998JA900165

class BiMax(object):
    def __init__(self, ant_len, ant_rad, base_cap):
        self.ant_len = ant_len
        self.ant_rad = ant_rad
        self.al_ratio = ant_rad / ant_len
        self.base_cap = base_cap
        # Unit conversion constants to deliver end results in SI units
        self.z_unit = 8.313797e6
        self.v_unit = 1.62760e-15

    @staticmethod
    def d_l(z, wc, n, t):
        """
        Longitudinal dispersion tensor
        z: w/kv_Tc
        wc: w/w_pc
        n: n_h/n_c
        t: T_h/T_c
    
        """
        return 1 - (z / wc) ** 2 * (zpd(z) + n / t * zpd(z / mp.sqrt(t)))

    @staticmethod
    def dz_dl(z, wc, n, t):
        """
        partial derivative of dl w.r.t z
        
        """
        zwc2 = (z / wc) ** 2
        nt = n / t
        sqt = mp.sqrt(t)
        zsqt = z / sqt
        result = -2 * zwc2 / z * (zpd(z) + nt * zpd(zsqt))
        result -= zwc2 * (zp2d(z) + nt / sqt * zp2d(zsqt))
        return result

    @staticmethod
    def dz2_dl(z, wc, n, t):
        """
        second order P.D. \frac{\partial^2 dl}{\partial z^2}
        The expansion was done by Mathematica.
        """
        z2 = z * z
        z4 = z2 * z2
        t2 = t * t
        t3 = t2 * t
        tsq = mp.sqrt(t)
        term_1 = 1 - 6 * z2 + 2 * z4
        term_2 = (n / t3) * (t2 - 6 * t * z2 + 2 * z4)
        term_3 = z * (3 - 7 * z2 + 2 * z4) * zp(z)
        term_4 = n * z / t3 / tsq * (3 * t2 - 7 * t * z2 + 2 * z4) * zp(z / tsq)

        return 4 * (term_1 + term_2 + term_3 + term_4) / wc ** 2

    @staticmethod
    def long_interval(w, n, t):
        """
        Return the interval of integration for longitudinal impedance, which
        is of the form [0, root, inf], where root satisfies
        d_l(root, w, n, t) = 0.
        
        """
        if w < mp.sqrt(1 + n):
            return [0, mp.inf]
        if n == 0:
            guesses = [4, 6, 8, 10]
        else:
            guesses = [4, 6, 8, 10, 12, 14]
        int_range = [0, mp.inf]
        for guess in guesses:
            try:
                root = mp.findroot(lambda z: BiMax.d_l(z, w, n, t).real, guess)
            except ValueError:
                continue
            unique = True
            for z in int_range:
                if mp.fabs(z - mp.fabs(root)) < 1e-4:
                    unique = False
            if unique:
                int_range += [mp.fabs(root)]
        int_range = np.sort(int_range)
        return int_range

    def za_l_integrand(self, z, wc, l, n, t):
        """
        Integrand of longitudinal component of the antenna impedance.
        l: l/l_d, where l is antenna length, l_d is debye length
        a: a/l is the ratio of antenna radius and monopole length
        
        """
        kl = wc * l / mp.sqrt(2) / z
        ka = kl * self.al_ratio
        num = f1(kl) * j0(ka) ** 2
        denom = z ** 2 * BiMax.d_l(z, wc, n, t)
        return num / denom

    def za_l(self, wc, l, n, t, tc):
        """
        Longitudinal impedance in unit of Ohms.
        wc: w/w_pc, where w_pc is core electron plasma frequency.
        tc: core electron temperature.

        """
        limits = self.long_interval(wc, n, t)
        # print(limits)
        result = mp.quad(lambda z: self.za_l_integrand(z, wc, l, n, t), limits)
        return result * self.z_unit * mp.mpc(0, 1) / mp.sqrt(tc)

    def bimax_integrand(self, z, wc, l, n, t):
        """
        Integrand of electron-noise integral.
        
        """
        return f1(wc * l / z / mp.sqrt(2)) * z * \
               (mp.exp(-z ** 2) + n / mp.sqrt(t) * mp.exp(-z ** 2 / t)) / \
               (mp.fabs(BiMax.d_l(z, wc, n, t)) ** 2 * wc ** 2)

    def bimax(self, wrel, l, n, t, tc):
        """
        electron noise.
        w: f/f_p, where f_p is the total plasma frequency.
        
        """
        wc = wrel * mp.sqrt(1 + n)
        limits = self.long_interval(wc, n, t)
        # print(limits)
        result = mp.quad(lambda z: self.bimax_integrand(z, wc, l, n, t), limits)
        return result * self.v_unit * mp.sqrt(tc)

    def zr_mp(self, wc, l, tc):
        """
        Return receiver impedance, assume mainly due to base capacitance.
        The antenna monopole length 45m is hardwired in.
        
        """
        ldc = self.ant_len / l
        nc = permittivity * boltzmann * tc / ldc ** 2 / echarge ** 2
        wpc = mp.sqrt(nc * echarge ** 2 / emass / permittivity)
        return mp.mpc(0, 1 / (wc * wpc * self.base_cap))

    def gamma_shot(self, wrel, l, n, t, tc):
        """
        Calculate 
        1, transfer gain
        2, electron shot noise
        
        """
        if wrel > 1.0 and wrel < 1.2:
            mp.mp.dps = 80
        else:
            mp.mp.dps = 40
        wc = wrel * mp.sqrt(1 + n)
        za = self.za_l(wc, l, n, t, tc)
        mp.mp.dps = 15
        zr = self.zr_mp(wc, l, tc)

        # below calculating shot noise:
        ldc = self.ant_len / l
        nc = permittivity * boltzmann * tc / ldc ** 2 / echarge ** 2
        vtc = np.sqrt(2 * boltzmann * tc / emass)
        ne = nc * vtc * (1 + n * mp.sqrt(t)) * 2 * np.pi * self.ant_rad * self.ant_len / np.sqrt(4 * np.pi)
        ###################
        ## a: coefficient in shot noise. see Issautier et al. 1999
        ###################
        a = 1 + echarge * 3.6 / boltzmann / tc
        shot_noise = 2 * a * echarge ** 2 * mp.fabs(za) ** 2 * ne
        return [mp.fabs((zr + za) / zr) ** 2, shot_noise]

    def proton(self, f, ne, n, t, tp, tc, vsw):
        """
        proton noise.
        wc: w/w_c, where w_c is the core electron plasma frequency.
        l: l_ant/l_dc, where l_ant is antenna length
        l_dc: core electron debye length.
        tep: T_e/T_p
        tc: core electron temperature
        vsw: solar wind speed.

        """
        ne = ne * 1e6
        tp = tp * echarge / boltzmann
        tc = tc * echarge / boltzmann
        w_p = np.sqrt(echarge ** 2 * ne / emass / permittivity)
        te = (tc + tc * t * n) / (1 + n)
        tg = tc * (1 + n) / (1 + n / t)
        ld = np.sqrt(permittivity * boltzmann * tg / ne / echarge ** 2)
        lrel = self.ant_len / ld
        vte = np.sqrt(2 * boltzmann * te / emass)
        omega = f * 2 * np.pi * ld / vsw
        tep = tg / tp
        M = vsw / vte
        integrand = lambda y: y * fperp(y * lrel) / (y ** 2 + 1 + omega ** 2) / (y ** 2 + 1 + omega ** 2 + tep)
        integral = scint.quad(integrand, 0, np.inf, epsrel=1.e-4)
        return mp.sqrt(2 * emass * boltzmann * tg) / (4 * mp.pi * permittivity * M) * integral[0]

    def electron_noise(self, f, ne, n, t, tp, tc, vsw):
        """
        a wrapper for bimax method.
        takes in raw argument and calculate the relative arguments.

        """
        ne *= 1e6
        nc = ne / (1 + n)
        tc *= echarge / boltzmann
        ldc = np.sqrt(permittivity * boltzmann * tc / nc / echarge ** 2)
        lc = self.ant_len / ldc
        w_p = np.sqrt(echarge ** 2 * ne / emass / permittivity)
        wrel = f * 2 * mp.pi / w_p
        return self.bimax(wrel, lc, n, t, tc)

    def gain_shot(self, f, ne, n, t, tp, tc, vsw):
        """
        a wrapper for gamma_shot method.
        
        """
        ne *= 1e6
        nc = ne / (1 + n)
        tc *= echarge / boltzmann
        ldc = np.sqrt(permittivity * boltzmann * tc / nc / echarge ** 2)
        lc = self.ant_len / ldc
        w_p = np.sqrt(echarge ** 2 * ne / emass / permittivity)
        wrel = f * 2 * mp.pi / w_p
        return self.gamma_shot(wrel, lc, n, t, tc)
