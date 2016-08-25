import mpmath as mp
import numpy as np
import scipy.integrate as scint
from mpmath import gamma
from .util import (zk, zp, zpd, j0, f1)
from .util import (boltzmann, emass, echarge, permittivity, cspeed, fperp)

class MaxKappa(object):
    def __init__(self, ant_len, ant_rad, base_cap, scpot=0):
        self.ant_len = ant_len
        self.ant_rad = ant_rad
        self.al_ratio = ant_rad/ant_len
        self.base_cap = base_cap

    @staticmethod
    def e_l(zc, wc, n, t, k):
        """
        longitudinal susceptibility tensor.
        zc:= w/(k v_tc)
        wc:= w/w_pc
        k:= kappa
        n: = nh/nc
        t: = th/tc
    
        """
        tau = mp.sqrt(2*k/ (2*k-3) / t)
        return 1 - (zc/wc)**2 * zpd(zc) \
            + 2 * n * (tau*zc/wc)**2 * (1-1/(2*k) + tau*zc * zk(tau*zc, k))

    @staticmethod
    def b(zc, n, t, k):
        """
    
        """
        tau = mp.sqrt(2*k/ (2*k-3) / t)
        b1 = zc * mp.exp(-zc**2)
        b2 = n * tau * zc * gamma(k+1)/gamma(k-1/2)/k**(3/2)
        b2 /= (1+ (zc * tau)**2 / k)**k
        return b1 + b2

    @staticmethod
    def int_interval(wrel, n, t, k):
        """
        find out the almost-singular point and
        divide the integration integrals.
        
        """
        if wrel < 1 or wrel > 1.3:
            return [0, mp.inf]
        else:
            guesses = [4, 6, 8, 10, 12, 14]
        wc = wrel * mp.sqrt(1+n)
        int_range = [0, mp.inf]
        for guess in guesses:
            try:
                root = mp.findroot(lambda zc: MaxKappa.e_l(zc, wc, n, t, k), guess)
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

    @staticmethod
    def maxkappa_integrand(zc, wc, lc, n, t, k):
        """
        integrand of electron noise integral.
        lc:= l/ldc
    
        """
        num = f1(wc*lc/mp.sqrt(2)/zc) * MaxKappa.b(zc, n, t, k)
        denom = mp.fabs(MaxKappa.e_l(zc, wc, n, t, k))**2
        return num/denom

    @staticmethod
    def maxkappa(wrel, lc, n, t, k, Tc):
        """
        integral in electron noise calculation.
        wrel:= w/w_ptot, where w_ptot is the total electron plasma frequency
        Tc:= core electron temperature
        
        """
        wc = wrel * mp.sqrt(1+n)
        int_range = MaxKappa.int_interval(wrel, n, t, k)
        coeff = 16 * emass/mp.pi**(3/2) /permittivity /wc**2 * mp.sqrt(2*boltzmann*Tc/emass)
        integral = mp.quad(lambda z: MaxKappa.maxkappa_integrand(z, wc, lc, n, t, k), int_range)
        return coeff * integral 

    def electron_noise(self, f, ne, n, t, tp, tc, k, vsw):
        """
        a wrapper of maxkappa method.

        """
        ne = ne * 1e6
        tc = tc * echarge/boltzmann
        w_p = np.sqrt(echarge**2 * ne/emass/permittivity)
        wrel = f * 2 * np.pi / w_p
        ldc = np.sqrt(permittivity * boltzmann * tc/ne/ echarge**2)
        lc = self.ant_len/ldc
        return self.maxkappa(wrel, lc, n, t, k, tc)
    
    def za_integrand(self, zc, wc, lc, n, t, k):
        """
        integrand of the antenna integral
        """
        kl = wc*lc/mp.sqrt(2)/zc
        ka = kl * self.al_ratio
        return f1(kl) * j0(ka)**2 / MaxKappa.e_l(zc, wc, n, t, k) / zc**2

    def za(self, wrel, lc, n, t, k, Tc):
        """
        antenna impedance
        
        """
        wc = wrel * mp.sqrt(1+n)
        coeff = 4*mp.mpc(0,1) / mp.pi**2 / permittivity
        coeff *= mp.sqrt(emass/2/boltzmann/Tc)
        int_range = MaxKappa.int_interval(wrel, n, t, k)
        integral = mp.quad(lambda zc: self.za_integrand(zc, wc, lc, n, t, k), int_range, method='tanh-sinh')
        return coeff * integral
    
        
    def zr(self, wc, lc, tc):
        """
        Return receiver impedance, assume mainly due to base capacitance.
        The antenna monopole length 45m is hardwired in.
        
        """
        ldc = self.ant_len/lc
        nc = permittivity * boltzmann * tc/ ldc**2 / echarge**2
        wpc = mp.sqrt( nc * echarge**2 / emass / permittivity)
        return mp.mpc(0, 1/(wc * wpc * self.base_cap))

    def impedance(self, f, ne, n, t, tp, tc, k, vsw):
        """
        returns antenna impedance and base capacitance impedance.
        """
        ne *= 1e6
        nc = ne/(1+n)
        tc *= echarge/ boltzmann
        ldc = np.sqrt(permittivity * boltzmann * tc/nc/echarge**2)
        lc = self.ant_len/ldc
        w_p = np.sqrt(echarge**2 * ne/emass/permittivity)
        wrel = f * 2 * mp.pi/w_p
        if wrel > 1.0 and wrel < 1.1:
            mp.mp.dps = 40
        else:
            mp.mp.dps = 20
        za_val = self.za(wrel, lc, n, t, k, tc)
        mp.mp.dps = 15
        wc = wrel * (1+n)
        zr_val = self.zr(wc, lc, tc)
        return [za_val, zr_val]

    def gamma_shot(self, wrel, l, n, t, k, tc):
        """
        Calculate 
        1, transfer gain
        2, electron shot noise
        
        """
        if wrel > 1.0 and wrel < 1.1:
            mp.mp.dps = 40
        else:
            mp.mp.dps = 20
        wc = wrel * mp.sqrt(1+n)
        za_val = self.za(wrel, l, n, t, k, tc)
        mp.mp.dps = 15
        zr_val = self.zr(wc, l, tc)

        # below calculating shot noise:
        ldc = self.ant_len/l
        nc  =permittivity * boltzmann * tc/ ldc**2 / echarge**2
        vtc = np.sqrt(2 * boltzmann * tc/ emass)
        ne = nc * vtc * (1 + n * mp.sqrt(t)) * 2 * np.pi * self.ant_rad * self.ant_len / np.sqrt(4 * np.pi)
        ###################################
        ## a: coefficient in shot noise. ##
        ###################################
        scpot= 4
        a = 1 + echarge * scpot / boltzmann/tc
        shot_noise = 2 * a * echarge**2 * mp.fabs(za_val)**2 * ne        
        return [mp.fabs((zr_val+za_val)/zr_val)**2, shot_noise]

    def proton(self, f, ne, n, t, tp, tc, k, vsw):
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
        tp = tp * echarge/boltzmann
        tc = tc * echarge/boltzmann
        w_p = np.sqrt(echarge**2 * ne/emass/permittivity)
        te = (tc + tc * t * n)/(1+n)
        tg = tc * (1 + n)/(1 + n/t)
        ld = np.sqrt(permittivity * boltzmann * tg/ne/ echarge**2)
        lrel = self.ant_len/ld
        vte = np.sqrt(2 * boltzmann * te/ emass)
        omega = f * 2 * np.pi * ld/vsw
        tep = tg/tp
        M = vsw/vte
        integrand = lambda y: y * fperp(y * lrel)/ (y**2 + 1 + omega**2) / (y**2 + 1 + omega**2 + tep)
        integral = scint.quad(integrand, 0, np.inf, epsrel = 1.e-4) 
        return mp.sqrt(2*emass*boltzmann*tg)/(4*mp.pi*permittivity* M) *  integral[0]

    def proton_parallel(self, f, ne, n, t, tp, tc, k, vsw):
        """
        proton noise when antenna is parallel to the plasma flow.

        """
        ne = ne * 1e6
        tp = tp * echarge/boltzmann
        tc = tc * echarge/boltzmann
        tg = tc * (1 + n)/(1 + n/t)
        ld = np.sqrt(permittivity * boltzmann * tg/ne/ echarge**2)
        lrel = self.ant_len/ld
        omega = f * 2 * np.pi * ld/vsw
        tep = tg/tp
        term_1 = 8 * boltzmann/ (np.pi * permittivity)
        term_2 = tg/vsw/tep
        term_3 = np.sin(lrel * omega/2)**4 / (lrel*omega)**2
        term_4 = np.log((1+t+omega**2)/(1+omega**2))
        return term_1 * term_2 * term_3 * term_4
        
