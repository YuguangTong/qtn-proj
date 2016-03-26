from qtn.bimax import BiMax
from qtn.util import f1, j0
from qtn.bimax_util import z_b
import numpy as np
import sympy.mpmath as mp
import matplotlib.pyplot as plt
#%matplotlib inline

ant_len = 50      # m (monopole) 
ant_rad = 1.9e-1  # m
base_cap = 20e-12 # Farad
al_ratio = ant_rad / ant_len

p = BiMax(ant_len, ant_rad, base_cap)

# integrand of V^2
def integrand(z, wc, l, n, t):
    return f1(wc*l/z/mp.sqrt(2)) * z * \
            (mp.exp(-z**2) + n/mp.sqrt(t)*mp.exp(-z**2 / t)) / \
            mp.fabs(BiMax.d_l(z, wc, n, t))**2 / wc**2
        
# integrand of V^2, with the bessel term included        
def integrand_2(z, wc, l, n, t):
    kl = wc*l/z/mp.sqrt(2)
    ka = kl * al_ratio
    j02 = j0(ka)**2
    #print(j02)
    return f1(kl) * j02 * z * \
            (mp.exp(-z**2) + n/mp.sqrt(t)*mp.exp(-z**2 / t)) / \
            mp.fabs(BiMax.d_l(z, wc, n, t))**2 / wc**2

wrel, l, n, t, tc = 0.1, 5, 0., 10, 1
wc = wrel * np.sqrt(1 + n)

# chunk of code that decide the location of near singularity of the integrand
if wrel < 1 or wrel > 1.2:
    z0 = 100
else:
    if wrel <1.05:
        guess = z_b(wc, n, t)
    else: 
        guess = z_b(wc, 0, t)
    print('guess = ', guess)
    z0 = mp.findroot(lambda z: BiMax.d_l(z, wc, n, t).real, guess)
print('z0 = ', z0)
print(BiMax.d_l(z0, wc, n, t))

z_arr = np.logspace(-8, 1, 100)
dl_arr = np.array([integrand(z, wc, l, n, t) for z in z_arr])
dl_arr_2 = np.array([integrand_2(z, wc, l, n, t) for z in z_arr])
plt.plot(z_arr, dl_arr)
plt.plot(z_arr, dl_arr_2)
#plt.xscale('log')
plt.yscale('log')
#plt.axvline(z0,linestyle = 'dashed')
#plt.axhline(0,linestyle = 'dashed')

z1 = 0
z2 = 5
if z0 < z2:
    limits = [z1, z0, z2]
else:
    limits = [z1, z2]
    
int1 = mp.quad(lambda z: integrand(z, wc, l, n, t), limits)
int2 = mp.quad(lambda z: integrand_2(z, wc, l, n, t), limits)
print(int1)
print(int2)
