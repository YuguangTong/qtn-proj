# utility functions for BiMax class
from qtn.util import zp_sp, zpd_sp
from sympy.mpmath import mp, fp
import numpy as np
from scipy.optimize import fsolve
from scipy.special import sici, j0

# fundamental constants
boltzmann = 1.3806488e-23  # J/K
emass = 9.10938291e-31     # kg
pmass = 1.67262178e-27
echarge = 1.60217657e-19   # C
permittivity = 8.854187817e-12  # F/m
cspeed = 299792458         # m/s

def zp(x):
    """
    plasma dispersion function                                
    using complementary error function in mpmath library.                       
                                                                                
    """
    
    #real = -fp.sqrt(fp.pi) * mp.exp(-x**2) * mp.erfi(x)
    #imag = fp.sqrt(fp.pi) * mp.exp(-x**2)

    return  mp.sqrt(mp.pi) * mp.exp(-x**2) * mp.mpc(-mp.erfi(x), '1')


def zpd(x):
    """
    Derivative of plasma dispersion function.                                                   
                                                                                
    """
    return -2  * (1 + x * zp(x))

def zp2d(x):
    """
    second derivative of plasma dispersion function.
    """
    return 4 * x - 2 * zp(x) + 4 * x**2 * zp(x)

def d_l_sp(z, wc, n, t):
    """
    Longitudinal dispersion tensor
    z: w/kv_Tc
    wc: w/w_pcdef f1(x):
    """

    return 1 - (z/wc)**2 * (zpd_sp(z) + n/t * zpd_sp(z/np.sqrt(t)))


def f1_sp(x):
    """
    scipy version of the function f1 (Kuehl 1966, Couturier 1981)
    """
    term1 = x * (sici(x)[0] - 0.5 * sici(2 * x)[0])
    term2 = - 2 * np.sin(0.5 * x)**4
    return (term1 + term2) / x**2

def f1(x):
    """
    An angular integral that appears in electron noise calculation.
    
    """
    term1 = x * (fp.si(x) - 0.5 * fp.si(2*x))
    term2 = -2 * fp.sin(0.5 * x)**4
    return (term1 + term2) / x**2

def j02(x):
    """
    A wrapper of Bessel function of the first kind,
    up to 1% accuracy.
    """
    if x < 0.1:
        return 1 - x**2 / 2
    if x > 1000:
        return (1 + fp.sin(2 * x)) / (fp.pi * x)
    return fp.besselj(0, x)**2

def d_l(z, wc, n, t):
    """
    Longitudinal dispersion tensor
    z: w/kv_Tc
    wc: w/w_pc
    n: n_h/n_c
    t: T_h/T_c

    """
    return 1 - (z/wc)**2 * (zpd(z) + n/t * zpd(z/fp.sqrt(t)))
    
def dz_dl(z, wc, n, t):
    """
    partial derivative of dl w.r.t z
        
    """
    zwc2 = (z/wc)**2
    nt = n/t
    sqt = fp.sqrt(t)
    zsqt = z/sqt
    result = -2 * zwc2 / z * (zpd(z) + nt * zpd(zsqt))
    result -= zwc2 * (zp2d(z) + nt/sqt * zp2d(zsqt))
    return result

def dz2_dl(z, wc, n, t):
    """
    second order P.D. \frac{\partial^2 dl}{\partial z^2}
    The expansion is done by Mathematica. 
    """
    z2 = z*z
    z4 = z2 * z2
    t2 = t * t
    t3 = t2 * t
    tsq = fp.sqrt(t)
    term_1 = 1 - 6 * z2 + 2 * z4
    term_2 = (n/t3) * (t2 - 6 * t * z2 + 2 * z4)
    term_3 = z * (3 - 7 * z2 + 2 * z4) * zp(z)
    term_4 = n * z / t3 / tsq * (3 * t2 - 7 * t * z2 + 2 * z4) * zp(z/tsq)
        
    return 4 * (term_1 + term_2 + term_3 + term_4) / wc**2

def z_b(wc, n, t):
    """
    the approximated expression for the pole,
    when frequency is close to plasma frequency,
    in two-Maxwellian-electron plasma.
    
    see Meyer-Vernet & Perche (1989), the "toolkit" paper
    """
    term_1 = wc/(fp.sqrt(2) * (1 + n))
    term_2 = 3 * (1 + n * t) / (1 - (1 + n) / wc**2 ) 
    if term_2 < 0:
        print('Specified frequency is lower than total plasma frequency...')
    term_2 = fp.sqrt(term_2)
    return term_1 * term_2    