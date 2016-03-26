from sympy.mpmath import mp, fp
from .bimax_util import z_b, f1, j02, d_l, dz_dl
# at this moment we do not do optimization. We will leave most of the calculation in mp context even though it could be done in fp context. 

class new_BiMax(object):
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
        Integrand of the electron noise integral.
        We leave out Bessel function in the expression. 
        Since when ant_rad / ant_len <~ 0.005, the relative error
        in electron noise < 1%. 
        """
        return f1(wc*l/z/fp.sqrt(2)) * z * \
            (fp.exp(-z**2) + n/fp.sqrt(t) * fp.exp(-z**2 / t)) / \
                mp.fabs(d_l(z, wc, n, t))**2 / wc**2    

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
            z0 = mp.findroot(lambda z: d_l(z, wc, n, t).real, guess)
            print('z0 = ', z0)
            return z0
        except Exception:
            return None
        
    def new_bimax(self, wrel, l, n, t, tc):
        """
        electron noise.
        w: f/f_p, where f_p is the total plasma frequency.
        
        """
        wc = wrel * fp.sqrt(1+n)
        
        # Only near plasma frequency does integrand peak near z0
        
        if wrel < 1 or wrel > 1.2:
            result = fp.quad(lambda z: self.bimax_integrand(z, wc, l, n, t), [0, mp.inf])
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
            return result * self.v_unit * mp.sqrt(tc)
         
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
        kl0 = wc*l/z0/mp.sqrt(2)
        ka0 = kl0 * self.al_ratio
        # num = f1(kl0) * j02(ka0) * z0 /wc**2 * \
        num = f1(kl0) * z0 /wc**2 * \
                (fp.exp(-z0**2) + n/fp.sqrt(t)* fp.exp(-z0**2 / t))
        fac = 2 * mp.atan(dz_el_re/el_img * dz)
        
        int_2 = fac * num / el_img / dz_el_re 
        
        return (int_1 + int_2 + int_3) * self.v_unit * mp.sqrt(tc)


        
