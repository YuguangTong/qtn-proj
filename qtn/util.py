import time
import mpmath as mp
from mpmath import mpf, mpc
from scipy.special import j0

# fundamental constants
boltzmann = 1.3806488e-23  # Joule/Kelvin
emass = 9.10938291e-31     # kg
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

def zp_mp(x):
    """
    plasma dispersion function                                
    using complementary error function in mpmath library.                       
                                                                                
    """
    return -mp.sqrt(mp.pi) * mp.exp(-x**2) * mp.erfi(x) + mpc(0, 1) * mp.sqrt(mp.pi) * mp.exp(-x**2)

def zpd_mp(x):
    """
    Derivative of plasma dispersion function.                                                   
                                                                                
    """
    return -mpf(2) * (mpf(1) + x * zp_mp(x))

def j0_mp(x):
    """zeroth order bessel function, the argument may be of "mpf" type.
    May be faster than general besselj(n, x) from mpmath library.
    Need profiler to see.
    """
    return j0(float(str(x)))

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
