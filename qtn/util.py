import time
import mpmath as mp
import numpy as np
from mpmath import mpf, mpc
from mpmath import hyp2f1, gamma
import scipy.special
from scipy.special import j1, itj0y0, sici

# fundamental constants
# (ygtong): Retrospectively, these constants could have been imported from scipy.constants
# but refactor could be painful ...
boltzmann = 1.3806488e-23  # J/K
emass = 9.10938291e-31  # kg
pmass = 1.67262178e-27
echarge = 1.60217657e-19  # C
permittivity = 8.854187817e-12  # F/m
cspeed = 299792458  # m/s


def timing(f):
    """
    time decorator
    """

    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('{0} function took {1:.3f} ms'.format(f.__name__, (time2 - time1) * 1000.0))
        return ret

    return wrap


def ldeb(ne, te):
    """ 
    Return the Debye length.
    ne: electron density
    te: electron temperature
    SI units
    ~ 69.01 * sqrt(te/ne)

    """

    return mp.sqrt(permittivity * boltzmann / echarge ** 2) * mp.sqrt(te / ne)


def fp(ne):
    """ 
    Return the plasma frequency. 
    ~ 8.98 * sqrt(ne)

    """

    return mp.sqrt(echarge ** 2 / emass / permittivity) / 2 / mp.pi * mp.sqrt(ne)


# Two arrays that appear in 8-pole Pade approximation to plasma dispersion relation function.
# See the technical report accompany the Fortran code WHAMP, Ronnmark (1985)
#   https://link.springer.com/article/10.1007/BF00214996
# Pade approximation is fast because it doesn't involve any special functions. But the error
#   might be large outside of certain area in the complex plane.
c_arr = np.array([
    2.237687789201900 - 1.625940856173727j,
    -2.237687789201900 - 1.625940856173727j,
    1.465234126106004 - 1.789620129162444j,
    -1.465234126106004 - 1.789620129162444j,
    0.8392539817232638 - 1.891995045765206j,
    -0.8392539817232638 - 1.891995045765206j,
    0.2739362226285564 - 1.941786875844713j,
    -0.2739362226285564 - 1.941786875844713j])

b_arr = np.array([-0.01734012457471826 - 0.04630639291680322j,
                  -0.01734012457471826 + 0.04630639291680322j,
                  -0.7399169923225014 + 0.8395179978099844j,
                  -0.7399169923225014 - 0.8395179978099844j,
                  5.840628642184073 + 0.9536009057643667j,
                  5.840628642184073 - 0.9536009057643667j,
                  -5.583371525286853 - 11.20854319126599j,
                  -5.583371525286853 + 11.20854319126599j])


def zp_pade(z):
    """
    Pade approximations to plasma dispersion function.
    
    Keyword arguments:
    z: dimensionless argument of the plasma dispersion function.
    
    Return the value of Zp(z) using Pade approximations.
    """
    return np.sum(b_arr / (z - c_arr))


def zp_sp(z):
    """
    Plasma dispersion function
    Utilize the Dawnson function, dawsn, in scipy.special module.
    Keyword arguments:
    z: dimensionless argument of the plasma dispersion function.
    
    Return the value of Zp(z)
    """
    return -2. * scipy.special.dawsn(z) + 1.j * np.sqrt(np.pi) * np.exp(- z ** 2)


def zpd_sp(x):
    """
    Derivative of the plasma dispersion function
    
    """
    return -2 * (1 + x * zp_sp(x))


def zp(x):
    """
    plasma dispersion function                                
    using complementary error function in mpmath library.                       
                                                                                
    """
    return -mp.sqrt(mp.pi) * mp.exp(-x ** 2) * mp.erfi(x) + mpc(0, 1) * mp.sqrt(mp.pi) * mp.exp(-x ** 2)


def zpd(x):
    """
    Derivative of plasma dispersion function.                                                   
                                                                                
    """
    return -mpf(2) * (mpf(1) + x * zp(x))


def zp2d(x):
    """
    second derivative of plasma dispersion function.
    """
    return 4 * x - 2 * zp(x) + 4 * x ** 2 * zp(x)


def j0(x):
    """zeroth order bessel function, the argument may be of "mpf" type.
    May be faster than general besselj(n, x) from mpmath library.
    Need profiler to see.
    """
    return scipy.special.j0(float(str(x)))


def eta(tc):
    """
    eta = v_{tc}/c, where v_tc is the thermal speed
    of core electrons whose temperature is tc (Kelvin)
    
    """
    return mp.sqrt(2 * boltzmann * tc / emass) / cspeed


def f1(x):
    """
    An angular integral that appears in electron noise calculation.
    See N. Meyer-Vernet et al. (1989)
        https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JA094iA03p02405
    
    """
    term1 = x * (mp.si(x) - 0.5 * mp.si(2 * x))
    term2 = -2 * mp.sin(0.5 * x) ** 4
    return (term1 + term2) / x ** 2


def f2(x):
    """
    Another angular integral in electron noise calculation.
    See N. Meyer-Vernet et al. (1989)
        https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/JA094iA03p02405

    """
    term1 = 2 * x ** 3 * (2 * mp.si(2 * x) - mp.si(x))
    term2 = 2 * x ** 2 * (mp.cos(2 * x) - mp.cos(x))
    term3 = x * (mp.sin(2 * x) - 2 * mp.sin(x)) - (mp.cos(2 * x) - 4 * mp.cos(x) + 3)
    return (term1 + term2 + term3) / (12 * x ** 2)


def f1_sp(x):
    """
    An angular integral that appears in electron noise calculation.
    Definition is the same as f1 above, but scipy functions are used for speed.
    
    """
    term1 = x * (sici(x)[0] - 0.5 * sici(2 * x)[0])
    term2 = -2 * np.sin(0.5 * x) ** 4
    return (term1 + term2) / x ** 2


def f2_sp(x):
    """
    Another angular integral in electron noise calculation.
    Definition is the same as f2 above, but scipy functions are used for speed.
    """
    term1 = 2 * x ** 3 * (2 * np.si(2 * x) - np.si(x))
    term2 = 2 * x ** 2 * (np.cos(2 * x) - p.cos(x))
    term3 = x * (np.sin(2 * x) - 2 * np.sin(x)) - (np.cos(2 * x) - 4 * np.cos(x) + 3)
    return (term1 + term2 + term3) / (12 * x ** 2)


def fperp(x):
    """
    result of angular integration in proton noise integration.
    See Issautier et al. (1999)
        https://agupubs.onlinelibrary.wiley.com/doi/pdf/10.1029/1998JA900165
    
    """
    return 8. / x * (2 * itj0y0(x)[0] - itj0y0(2 * x)[0] + j1(2 * x) - 2 * j1(x))


def zk(z, k):
    """
    modified dispersion function for Kappa distribution.
    See Mace and Hellberg (1995)
        https://aip.scitation.org/doi/10.1063/1.871296
    
    """
    i = mp.mpc(0, 1)
    coeff = i * (k + 0.5) * (k - 0.5) / (mp.sqrt(k ** 3) * (k + 1))
    return coeff * hyp2f1(1, 2 * k + 2, k + 2, (1 - z / (i * mp.sqrt(k))) / 2)
