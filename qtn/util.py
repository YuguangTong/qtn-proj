import time
import mpmath as mp
from mpmath import mpf, mpc
from mpmath import hyp2f1, gamma
import scipy.special as scsp
from scipy.special import j1, itj0y0

# fundamental constants
boltzmann = 1.3806488e-23  # J/K
emass = 9.10938291e-31     # kg
pmass = 1.67262178e-27
echarge = 1.60217657e-19   # C
permittivity = 8.854187817e-12  # F/m
cspeed = 299792458         # m/s


def timing(f):
    """
    time decorator
    """
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms', f.__name__, (time2-time1)*1000.0)
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
    
    return mp.sqrt(permittivity * boltzmann/ echarge**2) * mp.sqrt(te/ne)

def fp(ne):
    """ 
    Return the plasma frequency. 
    ~ 8.98 * sqrt(ne)

    """
    
    return mp.sqrt(echarge**2/emass/permittivity)/2/mp.pi * mp.sqrt(ne)

def zp(x):
    """
    plasma dispersion function                                
    using complementary error function in mpmath library.                       
                                                                                
    """
    return -mp.sqrt(mp.pi) * mp.exp(-x**2) * mp.erfi(x) + mpc(0, 1) * mp.sqrt(mp.pi) * mp.exp(-x**2)

def zpd(x):
    """
    Derivative of plasma dispersion function.                                                   
                                                                                
    """
    return -mpf(2) * (mpf(1) + x * zp(x))

def j0(x):
    """zeroth order bessel function, the argument may be of "mpf" type.
    May be faster than general besselj(n, x) from mpmath library.
    Need profiler to see.
    """
    return scsp.j0(float(str(x)))

def eta(tc):
    """
    eta = v_{tc}/c, where v_tc is the thermal speed
    of core electrons whose temperature is tc (Kelvin)
    
    """
    return mp.sqrt(2 * boltzmann * tc / emass)/ cspeed

def f1(x):
    """
    An angular integral that appears in electron noise calculation.
    
    """
    term1 = x * (mp.si(x) - 0.5 * mp.si(2*x))
    term2 = -2 * mp.sin(0.5 * x)**4
    return (term1 + term2)/x**2

def f2(x):
    """
    Another angular integral in electron noise calculation.
    
    """
    term1 = 2*x**3 * (2*mp.si(2*x)-mp.si(x))
    term2 = 2*x**2 * (mp.cos(2*x) - mp.cos(x))
    term3 = x * (mp.sin(2*x) - 2*mp.sin(x)) - (mp.cos(2*x) - 4* mp.cos(x)  + 3)
    return (term1 + term2 + term3)/(12 * x**2)

def fperp(x):
    """
    result of angular integration in proton noise integration.
    
    """
    return 8./x * (2*itj0y0(x)[0] - itj0y0(2*x)[0] + j1(2*x) - 2*j1(x))

def zk(z, k):
    """
    modified dispersion function for Kappa distribution.
    (Mace and Hellberg, 1995)
    
    """
    i = mp.mpc(0, 1)
    coeff = i * (k + 0.5) * (k-0.5) / (mp.sqrt(k**3) * (k+1))
    return coeff * hyp2f1(1, 2*k+2, k+2, (1-z/(i * mp.sqrt(k)))/2)
