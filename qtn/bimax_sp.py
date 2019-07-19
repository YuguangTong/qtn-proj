from mpmath import fp
from .bimax_util import (z_b, do_cprofile, f1_sp, fperp, j02_sp, d_l_sp,
                         dz_dl_sp, complex_quad,
                         boltzmann, emass, echarge, permittivity, cspeed)
import scipy.optimize, scipy.special, scipy.integrate


import numpy as np

# sqrt(pi)

sqrt_pi = 1.7724538509055159

# sqrt(2)

sqrt_2 = 1.4142135623730951

class BiMax_sp(object):
    def __init__(self, ant_len, ant_rad, base_cap):
        self.ant_len = ant_len
        self.ant_rad = ant_rad
        self.al_ratio = ant_rad/ant_len
        self.base_cap = base_cap
        self.z_unit = 8.313797e6
        self.v_unit = 1.62760e-15
        
        if self.al_ratio > 0.005:
            print('warning: thick antenna requires '+ 
                  'including Bessel functions in evaluating electron noise')
            
    def bimax_integrand_sp(self, z, wc, l, n, t):
        """
        parameter
        ---------
            z: w/k/v_thc, where v_thc -> sqrt(2 * k_B * T_c / m_e)
            wc: w/w_pc, where w_pc --> plasma frequency of core electrons
            l: l_ant / l_D, l_Dc --> Debye length of core electrons
            n: n_h / n_c; density ratio between hot and cold electrons
            t: t_h / t_c; temperature ratio
            
        return
        ------
            integrand of the electron noise integral
            
        notes
        -----
            We leave out Bessel function in the expression. 
            Since when ant_rad / ant_len <~ 0.005, the relative error
            in electron noise < 1%. 
        """
        return f1_sp(wc*l/z/sqrt_2) * z * \
            (np.exp(-z**2) + n/np.sqrt(t) * np.exp(-z**2 / t)) / \
                np.absolute(d_l_sp(z, wc, n, t))**2 / wc**2  

    def peak_sp(self, wrel, n, t):
        """
        Find z0 s.t. Re(d_l) = 0 at z = z0.
        This corresponds to a peak of the integrand, if exists.
        """
        wc = wrel * np.sqrt(1 + n)
        
        # Only near plasma frequency does integrand peak near z0
        
        if wrel < 1 or wrel > 1.2:
            return None
        
        # provide initial guess 
        
        if wrel <1.05:
            guess = z_b(wc, n, t)
        else:
            guess = z_b(wc, 0, t)
        #print('guess = ', guess)
        try: 
            sol = scipy.optimize.fsolve(lambda z: d_l_sp(z, wc, n, t).real, guess)
            z0 = sol[0]
            #print('z0 = ', z0, 'd_l.real = ', d_l_sp(z0, wc, n, t).real)
            return z0
        except Exception:
            return None
        
    def bimax_sp(self, wrel, l, n, t, tc):
        """
        parameters
        ----------
            z: w/k/v_thc, where v_thc -> sqrt(2 * k_B * T_c / m_e)
            wc: w/w_pc, where w_pc --> plasma frequency of core electrons
            l: l_ant / l_D, l_Dc --> Debye length of core electrons
            n: n_h / n_c; density ratio between hot and cold electrons
            t: t_h / t_c; temperature ratio
            tc: t_c; core/ cold electron temperature (K)
        return
        ------
            electron thermal noise
            
        notes
        -----
            
        """
        wc = wrel * np.sqrt(1+n)
        
        # Only near plasma frequency does integrand peak near z0
        
        if wrel < 1 or wrel > 1.2:
            result = scipy.integrate.quad(lambda z: self.bimax_integrand_sp(z, wc, l, n, t), 0, np.inf)
            # notice that scipy quad return [integral, estimated_error]
            return result[0] * self.v_unit * np.sqrt(tc) 
        
        # warn user of unexamined parameter region
        
        if wrel < 1.005:
            print('warning: evaluating electron noise ' +
                  'at frequency very close to w_pT. Needs to investigate error')

        # location of possible peak of integrand
        
        z0 = self.peak_sp(wrel, n, t)
        
        # if didn't find a possible peak, then evaluate integral directly
        
        if not z0:
            result = scipy.integrate.quad(lambda z: self.bimax_integrand_sp(z, wc, l, n, t), 0, np.inf)
            return result[0] * self.v_unit * np.sqrt(tc)
         
        # otherwise, evalute if this is a big peak
        
        dl_imag = np.fabs(d_l_sp(z0, wc, n, t).imag)

        # print('dl_imag = ', dl_imag)
        
        # if a small peak, evaluate directly
        
        if dl_imag > 1e-4:
            # print('direct evaluating integral when peak is small')
            #result = fp.quad(lambda z: self.bimax_integrand_sp(z, wc, l, n, t), [0, z0, fp.inf])
            result_1 = scipy.integrate.quad(lambda z: self.bimax_integrand_sp(z, wc, l, n, t), 0, z0)
            result_2 = scipy.integrate.quad(lambda z: self.bimax_integrand_sp(z, wc, l, n, t), z0, np.inf)
            result = result_1[0] + result_2[0]
            return result * self.v_unit * np.sqrt(tc)
        
        # A big peak --> split the ingetral into three parts
        # [0, z1], [z1, z0, z2], [z2, inf]
        
        z1 = z0 * 0.9999
        z2 = z0 * 1.0001
        dz = z0 * 0.0001
        
        # interval [0, z1] & [z2, inf]
        int_1 = scipy.integrate.quad(lambda z: self.bimax_integrand_sp(z, wc, l, n, t), 0, z1)
        int_3 = scipy.integrate.quad(lambda z: self.bimax_integrand_sp(z, wc, l, n, t), z2, fp.inf)
        int_1, int_3 = int_1[0], int_3[0]
        
        # interval [z1, z0, z2]
        el_img = np.fabs(d_l_sp(z0, wc, n, t).imag)
        dz_el_re = np.fabs(dz_dl_sp(z0, wc, n, t).real)
        
        kl0 = wc*l/z0/sqrt_2
        ka0 = kl0 * self.al_ratio
        
        num = f1_sp(kl0) * z0 * j02_sp(ka0) /wc**2 * \
                (np.exp(-z0**2) + n/np.sqrt(t)* np.exp(-z0**2 / t))
        fac = 2 * np.arctan(dz_el_re/el_img * dz)
        
        int_2 = fac * num / el_img / dz_el_re 
        
        return (int_1 + int_2 + int_3) * self.v_unit * fp.sqrt(tc)


        
    def za_l_sp(self, wrel, l, n, t, tc):
        """
        parameter
        ---------
            z: w/k/v_thc, where v_thc -> sqrt(2 * k_B * T_c / m_e)
            wrel: w/w_pT, where w_pT --> plasma frequency of all electrons
            l: l_ant / l_D, l_Dc --> Debye length of core electrons
            n: n_h / n_c; density ratio between hot and cold electrons
            t: t_h / t_c; temperature ratio
            tc: t_c; core/ cold electron temperature (K)
            
        return
        ------
            integrand of the electron noise integral
            
        notes
        -----
        """
        # wc: w/w_pc, where w_pc --> plasma frequency of core electrons
        wc = wrel * np.sqrt(1+n)
        
        klz = wc*l/sqrt_2
        kaz = klz * self.al_ratio
        
        def integrand_za_sp(z, wc, l, n, t):
            kl = klz / z
            ka = kaz / z
            num = f1_sp(kl) * j02_sp(ka)
            denom = z**2 * d_l_sp(z, wc, n, t)
            return num/denom      
        
        # devide (0, inf) into (0, 0.01) and (0.01, inf)
        
        def integrand_za_small_arg_sp(z, wc, l, n, t):
            """
            asymptotic expression for z << 1.
            """
            kl = klz /z
            ka = kaz / z
            f1_kl = f1_sp(kl)
            num = f1_kl * j02_sp(ka)
            
            # small argument expansion of e_l
            nt, wc2, z2 = n/t, wc**2, z**2
            el_re = 1 + 2 * (1 + nt) * z2 / wc2
            el_imag = 2 * sqrt_pi * z**3 / wc2 * (1 + nt/np.sqrt(t))
            denom = z2 * (el_re + 1j * el_imag)
            return num/denom   
        
        # integral on (0, 0.01), where we use the expression for small z
        
        small_z = 0.01
        za_1 = fp.quad(lambda z: integrand_za_small_arg_sp(z, wc, l, n, t), [0, small_z])
        
        # if w/w_pT not in (1, 1.2), peak is neglegible or directly integrable
        
        if wrel < 1 or wrel > 1.2:
            za_2 = fp.quad(lambda z: integrand_za_sp(z, wc, l, n, t), [small_z, fp.inf])
            result = za_1 + za_2
            return result * self.z_unit * 1j / np.sqrt(tc)   
        
        # else the peak may needs special attention
        
        # warn user of unexamined parameter region
        
        if wrel < 1.005:
            print('warning: evaluating electron noise ' +
                  'at frequency very close to w_pT. Needs to investigate error')

        # location of possible peak of integrand
        
        z0 = self.peak_sp(wrel, n, t)        

        # if didn't find a possible peak, then evaluate integral directly
        
        if not z0:
            print('did not find a solution zeta_0 for wrel = ', wrel)
            za_2 = fp.quad(lambda z: integrand_za_sp(z, wc, l, n, t), [small_z, fp.inf])
            result = za_1 + za_2
            return result * self.z_unit * 1j / np.sqrt(tc)      
        
        # otherwise, divide the (small_z, inf) into 
        # (small_z, z1), (z1, z0, z2), (z2, inf)
        # z1 = 0.999*z0, z2 = 1.001*z0
        
        z1, z2, dz = z0 * 0.999, z0 * 1.001, z0 * 0.001
        za_2 = fp.quad(lambda z: integrand_za_sp(z, wc, l, n, t), [small_z, z1]) + \
                fp.quad(lambda z: integrand_za_sp(z, wc, l, n, t), [z2, fp.inf]) 
        # Re[integral] on (z1, z0, z2) is time consuming and unimportant, so we don't evaluate it
        # we evaluate the Im[integral] on (z1, z0, z2)
        
        # first evalute if this is a big peak
        
        dl_imag = np.fabs(d_l_sp(z0, wc, n, t).imag)

        #print('dl_imag = ', dl_imag)
        
        # if a small peak, evaluate directly
        
        if dl_imag > 1e-4:
            
            print('direct evaluating integral when peak is small')
            za_3 = fp.quad(lambda z: integrand_za_sp(z, wc, l, n, t).imag, [z1, z0, z2])
            za_3 = fp.mpc(0, za_3)
        else:
            
            # if big peak, use analytic expression (approximate the peak by h(z) / (a^2 + b^2 z^2) )
            
            el_img = d_l_sp(z0, wc, n, t).imag
            el_img_abs = np.fabs(el_img)
            dz_el_re = np.fabs(dz_dl_sp(z0, wc, n, t).real)
            kl0 = klz / z0
            ka0 = kaz / z0
            num = f1_sp(kl0) * j02_sp(ka0) / z0**2 * (-el_img)
            fac = 2 * np.arctan(dz_el_re/el_img_abs * dz)
            za_3 = 1j* fac * num / el_img_abs / dz_el_re

        result = za_1 + za_2 + za_3
        return result * self.z_unit * 1j / np.sqrt(tc) 
    
    def zr(self, wc, l, tc):
        """
        parameter
        ---------
            wc: w/w_pc, where w_pc --> plasma frequency of core electrons
            l: l_ant / l_D, l_Dc --> Debye length of core electrons
            n: n_h / n_c; density ratio between hot and cold electrons
            tc: t_c; core/ cold electron temperature (K)

        return
        ------
            receiver impedance
            
        notes
        -----
            assume impedance mainly comes from base capacitance.
        
        """
        ldc = self.ant_len/l
        nc = permittivity * boltzmann * tc/ ldc**2 / echarge**2
        wpc = np.sqrt( nc * echarge**2 / emass / permittivity)
        return 1j/(wc * wpc * self.base_cap)  
     
          
    def gamma_sp(self, wrel, l, n, t, tc):
        """
        parameters
        ----------
            wrel: w/w_pT, where w_pT --> plasma frequency of all electrons
            l: l_ant / l_D, l_Dc --> Debye length of core electrons
            n: n_h / n_c; density ratio between hot and cold electrons
            t: t_h / t_c; temperature ratio
            tc: t_c; core/ cold electron temperature (K)

        return
        ------
            antenna gain Gamma = V_w^2 / V_a^2
            
        notes
        -----
            assume impedance mainly comes from base capacitance.
            see Couturier et al. (1981) 10.1029/JA086iA13p11127
        
        """
        wc = wrel * np.sqrt(1+n)
        za = self.za_l_sp(wrel, l, n, t, tc)
        zr = self.zr(wc, l, tc)                        
        return np.absolute((zr+za)/zr)**2 

    def gamma_shot_sp(self, wrel, l, n, t, tc):
        """
        parameters
        ----------
            wrel: w/w_pT, where w_pT --> plasma frequency of all electrons
            l: l_ant / l_D, l_Dc --> Debye length of core electrons
            n: n_h / n_c; density ratio between hot and cold electrons
            t: t_h / t_c; temperature ratio
            tc: t_c; core/ cold electron temperature (K)

        return
        ------
            [gamma, shot_noise]
            antenna gain Gamma = V_w^2 / V_a^2

            
        notes
        -----
            assume impedance mainly comes from base capacitance.
            see Couturier et al. (1981) 10.1029/JA086iA13p11127
        
        """
        wc = wrel * np.sqrt(1+n)
        za = self.za_l_sp(wrel, l, n, t, tc)
        zr = self.zr(wc, l, tc)                        
        gamma = np.absolute((zr+za)/zr)**2

        ldc = self.ant_len/l
        nc  =permittivity * boltzmann * tc/ ldc**2 / echarge**2
        vtc = np.sqrt(2 * boltzmann * tc/ emass)
        ne = nc * vtc * (1 + n * np.sqrt(t)) * 2 * np.pi * self.ant_rad * self.ant_len / np.sqrt(4 * np.pi)        
        #---------------------------------------------------------
        # a: coefficient in shot noise. see Issautier et al. 1999
        #---------------------------------------------------------
        # a = 1 + echarge * 3.6 / boltzmann/tc
        a = 1
        shot_noise = 2 * a * echarge**2 * np.abs(za)**2 * ne
        return [gamma, shot_noise]

    def proton_sp(self, f, ne, n, t, tp, tc, vsw, interp = False, interp_func = None):
        
        """
        parameters
        ----------
            f: frequency Hz
            ne: total electron density (cm^-3)
            n: n_h / n_c; density ratio between hot and cold electrons
            t: t_h / t_c; temperature ratio
            tp: proton temperature (eV)
            tc: t_c; core/ cold electron temperature (eV)
            vsw: solar wind velocity (km/s)

        keyword parameters
        ------------------
            interp: whether to calculate the integral by interpolation
                    if True, a interpolation function must be supplied by "inter_fn"
            interp_func: interpolation function to speed up calculation
            
        return
        ------
            proton thermal noise power density (V^2 / Hz)          
            
        notes
        -----
            see Issautier et al. (1999) 10.1029/1998JA900165
        
        """
        # change to SI unit
        ne = ne * 1e6
        tp, tc = tp * 11604.519339023796, tc * 11604.519339023796
        #te = tc * (1 + t * n) / (1 + n)
        
        # effective temperature 
        tg = tc * (1 + n)/(1 + n/t)
        
        # debye length
        # ld = np.sqrt(permittivity * boltzmann * tg/ne/ echarge**2)
        ld = 69.008978452135807 * np.sqrt(tg / ne)
        
        lrel = self.ant_len / ld
        
        # core electron thermal speed
        # vte = np.sqrt(2 * boltzmann * tc/ emass)
        vte = 5505.6947431410626 * np.sqrt(tc)
        
        # dimensionless params in the integral
        omega = f * 2 * np.pi * ld/vsw
        tep = tg/tp
        M = vsw/vte
        
        if interp:
            integral = interp_func(lrel, omega, tep)
        else:
            integrand = lambda y: \
                y * fperp(y * lrel)/ (y**2 + 1 + omega**2) / (y**2 + 1 + omega**2 + tep)
            integral = scipy.integrate.quad(integrand, 0, np.inf, epsrel = 1.e-4) 
        
        coeff = np.sqrt(2*emass*boltzmann * tg)/(4*np.pi*permittivity * M)
        # coeff = 4.5075701323600409e-17 * np.sqrt(tg) / M 
        return coeff *  integral[0]        
    
    def electron_noise_sp(self, f, ne, n, t, tp, tc, vsw):
        """
        a wrapper for bimax_sp method.
        takes in raw argument and calculate the relative arguments.

        """
        ne *= 1e6
        nc = ne/(1+n)
        tc *= echarge/ boltzmann
        ldc = np.sqrt(permittivity * boltzmann * tc/nc/echarge**2)
        lc = self.ant_len/ldc
        w_p = np.sqrt(echarge**2 * ne/emass/permittivity)
        wrel = f * 2 * np.pi/w_p
        return self.bimax_sp(wrel, lc, n, t, tc)

    def gain_shot_sp(self, f, ne, n, t, tp, tc, vsw):
        """
        a wrapper for gamma_shot method.
        
        """
        ne *= 1e6
        nc = ne/(1+n)
        tc *= echarge/ boltzmann
        ldc = np.sqrt(permittivity * boltzmann * tc/nc/echarge**2)
        lc = self.ant_len/ldc
        w_p = np.sqrt(echarge**2 * ne/emass/permittivity)
        wrel = f * 2 * np.pi/w_p
        return self.gamma_shot_sp(wrel, lc, n, t, tc)
