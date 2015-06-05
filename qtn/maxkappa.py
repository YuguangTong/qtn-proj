import mpmath as mp
import numpy as np
import scipy.integrate as scint
from mpmath import gamma
from .util import (zk, zp, zpd, j0, f1)
from .util import (boltzmann, emass, echarge, permittivity, cspeed)

class MaxKappa(object):
    def __init__(self, ant_len, ant_rad, base_cap):
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
        if wrel < 1:
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
    def bimax_integrand(self, z, wc, l, n, t):
        """
        Integrand of electron-noise integral.
        
        """
        return f1(wc*l/z/mp.sqrt(2)) * z * \
            (mp.exp(-z**2) + n/mp.sqrt(t)*mp.exp(-z**2 / t)) / \
            (mp.fabs(BiMax.d_l(z, wc, n, t))**2 * wc**2)

    @staticmethod
    def bimax(self, wrel, l, n, t, tc):
        """
        electron noise.
        w: f/f_p, where f_p is the total plasma frequency.
        
        """
        wc = wrel * mp.sqrt(1+n)
        limits = self.long_interval(wc, n, t)
        #print(limits)
        result = mp.quad(lambda z: self.bimax_integrand(z, wc, l, n, t), limits)
        return result * self.v_unit * mp.sqrt(tc)

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
        integral = mp.quad(lambda zc: self.za_integrand(zc, wc, lc, n, t, k), int_range)
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
        wc = wrel * mp.sqrt(1+n)
        za_val = self.za(wrel, l, n, t, tc)
        mp.mp.dps = 15
        zr_val = self.zr(wc, l, tc)

        # below calculating shot noise:
        ldc = self.ant_len/l
        nc  =permittivity * boltzmann * tc/ ldc**2 / echarge**2
        vtc = np.sqrt(2 * boltzmann * tc/ emass)
        ne = nc * vtc * (1 + n * mp.sqrt(t)) * 2 * np.pi * self.ant_rad * self.ant_len / np.sqrt(4 * np.pi)
        ###################
        ## a: coefficient in shot noise. see Issautier et al. 1999
        ###################
        a = 1 + echarge * 3.6 / boltzmann/tc
        shot_noise = 2 * a * echarge**2 * mp.fabs(za_val)**2 * ne        
        return [mp.fabs((zr_val+za_val)/zr_val)**2, shot_noise]

    def proton(self, wc, l, t, tep, tc, vsw):
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
        integral = scint.quad(integrand, 0, np.inf, epsrel = 1.e-8) 
        return integral[0] * boltzmann * tc/ (2 * np.pi * permittivity * vsw)
 
