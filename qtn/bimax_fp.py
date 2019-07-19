from mpmath import mp, fp
from .bimax_util import (z_b, f1, j02, d_l, dz_dl, do_cprofile,
                         boltzmann, emass, echarge, permittivity, cspeed)
# at this moment we do not do optimization. We will leave most of the calculation in mp context even though it could be done in fp context. 

class BiMax_fp(object):
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
            
    def bimax_integrand(self, z, wc, l, n, t):
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
        return f1(wc*l/z/fp.sqrt(2)) * z * \
            (fp.exp(-z**2) + n/fp.sqrt(t) * fp.exp(-z**2 / t)) / \
                fp.fabs(d_l(z, wc, n, t))**2 / wc**2    

    def peak(self, wrel, n, t):
        """
        Find z0 s.t. Re(d_l) = 0 at z = z0.
        This corresponds to a peak of the integrand, if exists.
        """
        wc = wrel * fp.sqrt(1+n)
        
        # Only near plasma frequency does integrand peak near z0
        
        if wrel < 1 or wrel > 1.2:
            return None
        
        # provide initial guess 
        
        if wrel <1.05:
            guess = z_b(wc, n, t)
        else:
            guess = z_b(wc, 0, t)
        print('guess = ', guess)
        try: 
            z0 = fp.findroot(lambda z: d_l(z, wc, n, t).real, guess)
            print('z0 = ', z0)
            return z0
        except Exception:
            return None
        
    #@do_cprofile     
    def bimax_fp(self, wrel, l, n, t, tc):
        """
        parameters
        ----------
            z: w/k/v_thc, where v_thc -> sqrt(2 * k_B * T_c / m_e)
            wc: w/w_pc, where w_pc --> plasma frequency of core electrons
            l: l_ant / l_D, l_Dc --> Debye length of core electrons
            n: n_h / n_c; density ratio between hot and cold electrons
            t: t_h / t_c; temperature ratio
            tc: t_c; core/ cold electron temperature
        return
        ------
            electron thermal noise
            
        notes
        -----
            
        """
        wc = wrel * fp.sqrt(1+n)
        
        # Only near plasma frequency does integrand peak near z0
        
        if wrel < 1 or wrel > 1.2:
            result = fp.quad(lambda z: self.bimax_integrand(z, wc, l, n, t), [0, fp.inf])
            return result * self.v_unit * fp.sqrt(tc)
        
        # warn user of unexamined parameter region
        
        if wrel < 1.005:
            print('warning: evaluating electron noise ' +
                  'at frequency very close to w_pT. Needs to investigate error')

        # location of possible peak of integrand
        
        z0 = self.peak(wrel, n, t)
        
        # if didn't find a possible peak, then evaluate integral directly
        
        if not z0:
            result = fp.quad(lambda z: self.bimax_integrand(z, wc, l, n, t), [0, fp.inf])
            return result * self.v_unit * fp.sqrt(tc)
         
        # otherwise, evalute if this is a big peak
        
        dl_imag = fp.fabs(d_l(z0, wc, n, t).imag)

        print('dl_imag = ', dl_imag)
        
        # if a small peak, evaluate directly
        
        if dl_imag > 1e-4:
            print('direct evaluating integral when peak is small')
            result = fp.quad(lambda z: self.bimax_integrand(z, wc, l, n, t), [0, z0, fp.inf])
            return result * self.v_unit * fp.sqrt(tc)
        
        # A big peak --> split the ingetral into three parts
        # [0, z1], [z1, z0, z2], [z2, inf]
        
        z1 = z0 * 0.9999
        z2 = z0 * 1.0001
        dz = z0 * 0.0001
        
        # interval [0, z1] & [z2, inf]
        int_1 = fp.quad(lambda z: self.bimax_integrand(z, wc, l, n, t), [0, z1])
        int_3 = fp.quad(lambda z: self.bimax_integrand(z, wc, l, n, t), [z2, fp.inf])

        # interval [z1, z0, z2]
        el_img = fp.fabs(d_l(z0, wc, n, t).imag)
        dz_el_re = fp.fabs(dz_dl(z0, wc, n, t).real)
        kl0 = wc*l/z0/fp.sqrt(2)
        ka0 = kl0 * self.al_ratio
        # num = f1(kl0) * j02(ka0) * z0 /wc**2 * \
        num = f1(kl0) * z0 /wc**2 * \
                (fp.exp(-z0**2) + n/fp.sqrt(t)* fp.exp(-z0**2 / t))
        fac = 2 * fp.atan(dz_el_re/el_img * dz)
        
        int_2 = fac * num / el_img / dz_el_re 
        
        return (int_1 + int_2 + int_3) * self.v_unit * fp.sqrt(tc)


        
    def za_l(self, wrel, l, n, t, tc):
        """
        parameter
        ---------
            z: w/k/v_thc, where v_thc -> sqrt(2 * k_B * T_c / m_e)
            wrel: w/w_pT, where w_pT --> plasma frequency of all electrons
            l: l_ant / l_D, l_Dc --> Debye length of core electrons
            n: n_h / n_c; density ratio between hot and cold electrons
            t: t_h / t_c; temperature ratio
            
        return
        ------
            integrand of the electron noise integral
            
        notes
        -----
        """
        # wc: w/w_pc, where w_pc --> plasma frequency of core electrons
        wc = wrel * fp.sqrt(1+n)
        
        klz = wc*l/fp.sqrt(2)
        kaz = klz * self.al_ratio
        
        def integrand_za(z, wc, l, n, t):
            kl = klz / z
            ka = kaz / z
            num = f1(kl) * j02(ka)
            denom = z**2 * d_l(z, wc, n, t)
            return num/denom      
        
        # devide (0, inf) into (0, 0.01) and (0.01, inf)
        
        def integrand_za_small_arg(z, wc, l, n, t):
            """
            asymptotic expression for z << 1.
            """
            kl = klz /z
            ka = kaz / z
            f1_kl = f1(kl)
            num = f1_kl * j02(ka)
            
            # small argument expansion of e_l
            nt, wc2, z2 = n/t, wc**2, z**2
            el_re = 1 + 2 * (1+nt) * z2 / wc2
            el_imag = 2 * fp.sqrt(fp.pi) * z**3 / wc2 * (1 + nt/fp.sqrt(t))
            denom = z2 * fp.mpc(el_re, el_imag)
            return num/denom   
        
        # integral on (0, 0.01), where we use the expression for small z
        
        small_z = 0.01
        za_1 = fp.quad(lambda z: integrand_za_small_arg(z, wc, l, n, t), [0, small_z])
        
        # if w/w_pT not in (1, 1.2), peak is neglegible or directly integrable
        
        if wrel < 1 or wrel > 1.2:
            za_2 = fp.quad(lambda z: integrand_za(z, wc, l, n, t), [small_z, fp.inf])
            result = za_1 + za_2
            return result * self.z_unit * fp.mpc(0, 1) / fp.sqrt(tc)   
        
        # else the peak may needs special attention
        
        # warn user of unexamined parameter region
        
        if wrel < 1.005:
            print('warning: evaluating electron noise ' +
                  'at frequency very close to w_pT. Needs to investigate error')

        # location of possible peak of integrand
        
        z0 = self.peak(wrel, n, t)        

        # if didn't find a possible peak, then evaluate integral directly
        
        if not z0:
            print('did not find a solution zeta_0 for wrel = ', wrel)
            za_2 = fp.quad(lambda z: integrand_za(z, wc, l, n, t), [small_z, fp.inf])
            result = za_1 + za_2
            return result * self.z_unit * fp.mpc(0, 1) / fp.sqrt(tc)      
        
        # otherwise, divide the (small_z, inf) into 
        # (small_z, z1), (z1, z0, z2), (z2, inf)
        # z1 = 0.999*z0, z2 = 1.001*z0
        
        z1, z2, dz = z0 * 0.999, z0 * 1.001, z0 * 0.001
        za_2 = fp.quad(lambda z: integrand_za(z, wc, l, n, t), [small_z, z1]) + \
                fp.quad(lambda z: integrand_za(z, wc, l, n, t), [z2, fp.inf]) 
        
        # Re[integral] on (z1, z0, z2) is time consuming and unimportant, so we don't evaluate it
        # we evaluate the Im[integral] on (z1, z0, z2)
        
        # first evalute if this is a big peak
        
        dl_imag = fp.fabs(d_l(z0, wc, n, t).imag)

        print('dl_imag = ', dl_imag)
        
        # if a small peak, evaluate directly
        
        if dl_imag > 1e-4:
            
            print('direct evaluating integral when peak is small')
            za_3 = fp.quad(lambda z: integrand_za(z, wc, l, n, t).imag, [z1, z0, z2])
            za_3 = fp.mpc(0, za_3)
            
        else:
            
            # if big peak, use analytic expression (approximate the peak by h(z) / (a^2 + b^2 z^2) )
            
            el_img = d_l(z0, wc, n, t).imag
            el_img_abs = fp.fabs(el_img)
            dz_el_re = fp.fabs(dz_dl(z0, wc, n, t).real)
            kl0 = klz / z0
            ka0 = kaz / z0
            num = f1(kl0) * j02(ka0) / z0**2 * (-el_img)
            fac = 2 * fp.atan(dz_el_re/el_img_abs * dz)
            za_3 = fp.mpc(0, fac * num / el_img_abs / dz_el_re)

        result = za_1 + za_2 + za_3
        return result * self.z_unit * fp.mpc(0, 1) / fp.sqrt(tc) 
    
    def zr(self, wc, l, tc):
        """
        parameter
        ---------
            wc: w/w_pc, where w_pc --> plasma frequency of core electrons
            l: l_ant / l_D, l_Dc --> Debye length of core electrons
            n: n_h / n_c; density ratio between hot and cold electrons
            tc: t_c; core/ cold electron temperature

        return
        ------
            receiver impedance
            
        notes
        -----
            assume impedance mainly comes from base capacitance.
        
        """
        ldc = self.ant_len/l
        nc = permittivity * boltzmann * tc/ ldc**2 / echarge**2
        wpc = fp.sqrt( nc * echarge**2 / emass / permittivity)
        return fp.mpc(0, 1/(wc * wpc * self.base_cap))  
     
          
    def gamma(self, wrel, l, n, t, tc):
        """
        parameters
        ----------
            wrel: w/w_pT, where w_pT --> plasma frequency of all electrons
            l: l_ant / l_D, l_Dc --> Debye length of core electrons
            n: n_h / n_c; density ratio between hot and cold electrons
            t: t_h / t_c; temperature ratio
            tc: t_c; core/ cold electron temperature

        return
        ------
            antenna gain Gamma = V_a^2 / V_w^2 ?
            
        notes
        -----
            assume impedance mainly comes from base capacitance.
        
        """
        wc = wrel * fp.sqrt(1+n)
        za = self.za_l(wrel, l, n, t, tc)
        zr = self.zr(wc, l, tc)                        
        return fp.fabs((zr+za)/zr)**2 
                        
