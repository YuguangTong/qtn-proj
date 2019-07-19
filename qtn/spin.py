
# calculate proton noise with angular dependence.

import numpy as np
import scipy.integrate
from qtn.util import timing, boltzmann, emass, echarge, permittivity, cspeed

def phi_integrand(y, beta, phi, omega, lrel):
    """
    integrand for the phi integral.
    
    Key parameters
    --------------
    y: dimensionless parameter for the outer integral
    beta: angle between antenna and solar wind velocity
    phi: azimuthal angle
    lrel: antenna length/debye length
    
    Return
    ------
    return the integrand for the phi integral
    """
    kl_cos_gamma = lrel * (omega * np.cos(beta) + y * np.sin(beta) * np.cos(phi))
    return np.sin(0.5 * kl_cos_gamma)**4 / kl_cos_gamma**2

def phi_integral(y, beta, omega, lrel):
    """
    value of the phi integral
    
    """
    return scipy.integrate.quad(lambda phi: phi_integrand(y, beta, phi, omega, lrel), 0, 2*np.pi)[0]


def proton_angle(beta, f, ne, n, t, tp, tc, k, vsw, ant_len):
    """
    proton noise at an arbitrary angle between antenna and solar wind velocity
    
    Key parameters
    --------------
    
    """
    ne = ne * 1e6
    tp = tp * echarge/boltzmann
    tc = tc * echarge/boltzmann
    w_p = np.sqrt(echarge**2 * ne/emass/permittivity)
    te = (tc + tc * t * n)/(1+n)
    tg = tc * (1 + n)/(1 + n/t)
    ld = np.sqrt(permittivity * boltzmann * tg/ne/ echarge**2)
    lrel = ant_len/ld
    vte = np.sqrt(2 * boltzmann * te/ emass)
    omega = f * 2 * np.pi * ld/vsw
    tep = tg/tp
    M = vsw/vte
    
    integrand = lambda y: y * phi_integral(y, beta, omega, lrel) / (1 + y**2 + omega**2) / (1 + tep + y**2 + omega**2)
    integral = scipy.integrate.quad(integrand, 0, np.inf)[0]
    coeff = 4 * np.sqrt(2 * emass * boltzmann * tg)/(np.pi**2 * permittivity * M)
    return coeff * integral
