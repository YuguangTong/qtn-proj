from qtn.bimax import BiMax
from qtn.util import f1, j0
from qtn.bimax_util import z_b
import numpy as np
import sympy.mpmath as mp
import matplotlib.pyplot as plt
%matplotlib inline

ant_len = 50      # m (monopole) 
ant_rad = 1.9e-4  # m
base_cap = 20e-12 # Farad
al_ratio = ant_rad / ant_len

# integrand of V^2, with the bessel term included        
def integrand_za(z, wc, l, n, t):
    kl = wc*l/mp.sqrt(2)/z
    ka = kl * al_ratio
    num = f1(kl) * j0(ka)**2
    denom = z**2 * BiMax.d_l(z, wc, n, t)
    return num/denom

def integrand_za_small_arg(z, wc, l, n, t):
    kl = wc*l/mp.sqrt(2)/z
    ka = kl * al_ratio
    #f1_kl = mp.pi/kl/4
    f1_kl = f1(kl)
    num = f1_kl * j0(ka)**2
    el_re = 1 + 2 * (1+n/t)*z**2 / wc**2
    el_imag = 2*mp.sqrt(mp.pi) * z**3 / wc**2 * (1 + n/t/mp.sqrt(t))
    denom = z**2 * mp.mpc(el_re, el_imag)
    return num/denom

wrel, l, n, t, tc = 1.1, 5, 0., 10, 1
wc = wrel * np.sqrt(1 + n)

z_arr = np.logspace(-12, 1.5, 100)
dl_arr = np.array([integrand_za(z, wc, l, n, t).real for z in z_arr])
dl_arr_2 = np.array([integrand_za_small_arg(z, wc, l, n, t).real for z in z_arr])
diff = np.array([mp.fabs(d) for d in (dl_arr - dl_arr_2)])
plt.plot(z_arr, dl_arr)
plt.plot(z_arr, dl_arr_2)
plt.plot(z_arr, diff/dl_arr)
plt.xscale('log')
plt.yscale('log')
#plt.axvline(z0,linestyle = 'dashed')
plt.axhline(0.01,linestyle = 'dashed')
plt.ylim([1e-10,1e5])

z0 = 100
z1 = 0
z2 = 0.0001
if z0 < z2:
    limits = [z1, z0, z2]
else:
    limits = [z1, z2]
    
int1 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits)
int2 = mp.quad(lambda z: integrand_za_small_arg(z, wc, l, n, t), limits)
print(int1)
print(int2)

