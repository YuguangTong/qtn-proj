import mpmath as mp
import numpy as np
from .util import (zp, zpd, j0, f1, f2)
from .util import (boltzmann, emass, echarge, permittivity, cspeed)

class BiMax(object):
    def __init__(self, ant_len, al_ratio, base_cap):
        self.ant_len = ant_len
        self.al_ratio = al_ratio
        self.base_cap = base_cap
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
        return 1 - (z/wc)**2 * (zpd(z) + n/t * zpd(z/mp.sqrt(t)))

    @staticmethod
    def long_interval(w, n, t):
        """
        Return the interval of integration for longitudinal impedance, which
        is of the form [0, root, inf], where root satisfies
        d_l(root, w, n, t) = 0.
        
        """
        if w < mp.sqrt(1+n):
            return [0, mp.inf]
        if n == 0:
            guesses = [4, 6, 8, 10]
        else:
            guesses = [4, 6, 8, 10, 12, 14]
        int_range = [0, mp.inf]    
        for guess in guesses:
            try:
                root = mp.findroot(lambda z: BiMax.d_l(z, w, n, t), guess)
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
        kl = wc*l/mp.sqrt(2)/z
        ka = kl * self.al_ratio
        num = f1(kl) * j0(ka)**2
        denom = z**2 * BiMax.d_l(z, wc, n, t)
        return num/denom

    def za_l(self, wc, l, n, t, tc):
        """
        Longitudinal impedance in unit of Ohms.
        wc: w/w_pc, where w_pc is core electron plasma frequency.
        tc: core electron temperature.

        """
        limits = self.long_interval(wc, n, t)
        print(wc, limits)
        result = mp.quad(lambda z: self.za_l_integrand(z, wc, l, n, t), limits)
        return result * self.z_unit * mp.mpc(0, 1)/ mp.sqrt(tc)

    def bimax_integrand(z, wc, l, n, t):
        """
        Integrand of electron-noise integral.
        
        """
        return f1(wc*l/z/mp.sqrt(2)) * z * \
            (mp.exp(-z**2) + n/mp.sqrt(t)*mp.exp(-z**2 / t)) / \
            (mp.fabs(d_l(z, wc, n, t))**2 * wc**2)

    def bimax(wrel, l, n, t, tc):
        """
        electron noise.
        w: f/f_p, where f_p is the total plasma frequency.
        
        """
        wc = wrel * mp.sqrt(1+n)
        limits = self.long_interval(wc, n, t)
        print(limits)
        result = mp.quad(lambda z: self.bimax_integrand(z, wc, l, n, t), limits)
        return result * self.v_unit * mp.sqrt(tc)

    def zr_mp(self, wc, l, tc):
        """
        Return receiver impedance, assume mainly due to base capacitance.
        The antenna monopole length 45m is hardwired in.
        
        """
        ldc = self.ant_len/l
        nc = permittivity * boltzmann * tc/ ldc**2 / echarge**2
        wpc = mp.sqrt( nc * echarge**2 / emass / permittivity)
        return mp.mpc(0, 1/(wc * wpc * self.base_cap))

    def gamma(self, wrel, l, n, t, tc):
        if wrel > 1.0 and wrel < 1.2:
            mp.mp.dps = 80
        else:
            mp.mp.dps = 40
        wc = wrel * mp.sqrt(1+n)
        za = self.za_l(wc, l, n, t, tc)
        zr = self.zr_mp(wc, l, tc)
        mp.mp.dps = 15
        return mp.fabs((zr+za)/zr)**2
