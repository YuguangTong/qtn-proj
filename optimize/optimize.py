from qtn.util import zpd_sp
import numpy as np

def d_l_sp(z, wc, n, t):
    """
        Longitudinal dispersion tensor
        z: w/kv_Tc
        wc: w/w_pc
        n: n_h/n_c
        t: T_h/T_c
        
        use scipy instead of mpmath
    """
    return 1 - (z/wc)**2 * (zpd_sp(z) + n/t * zpd_sp(z/np.sqrt(t)))
