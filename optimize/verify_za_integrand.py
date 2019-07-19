from qtn.bimax import BiMax
from qtn.util import f1, j0
from qtn.bimax_util import z_b
import numpy as np
import sympy.mpmath as mp
import matplotlib.pyplot as plt

ant_len = 50      # m (monopole) 
ant_rad = 1.9e-4  # m
base_cap = 20e-12 # Farad
al_ratio = ant_rad / ant_len

# integrand of Za
def integrand_za(z, wc, l, n, t):
    kl = wc*l/mp.sqrt(2)/z
    ka = kl * al_ratio
    num = f1(kl) * j0(ka)**2
    denom = z**2 * BiMax.d_l(z, wc, n, t)
    return num/denom

# approximating the integrand of Za
def integrand_za_approx(z0, z, wc, l, n, t):
    el_img = mp.fabs(BiMax.d_l(z0, wc, n, t).imag)
    dz_el_re = BiMax.dz_dl(z0, wc, n, t).real
    kl = wc*l/mp.sqrt(2)/z
    ka = kl * al_ratio
    num = f1(kl) * j0(ka)**2
    denom = z**2 * mp.mpc(dz_el_re * (z-z0), el_img)
    return num / denom
    

wrel, l, n, t, tc = 1.01, 5, 0., 1, 1
wc = wrel * np.sqrt(1 + n)

# chunk of code that decide the location of near singularity of the integrand
if wrel < 1 or wrel > 1.2:
    z0 = None
else:
    if wrel <1.05:
        guess = z_b(wc, n, t)
    else:
        guess = z_b(wc, 0, t)
    print('guess = ', guess)
    z0 = mp.findroot(lambda z: BiMax.d_l(z, wc, n, t).real, guess)
print('z0 = ', z0)
if z0:
    print(BiMax.d_l(z0, wc, n, t))

z_arr = z0 * np.linspace(0.999, 1.001, 100)

za_arr_1_re = np.array([integrand_za(z, wc, l, n, t).real for z in z_arr])
za_arr_1_imag = np.array([integrand_za(z, wc, l, n, t).imag for z in z_arr])
za_arr_2_re = np.array([integrand_za_approx(z0, z, wc, l, n, t).real for z in z_arr])
za_arr_2_imag = np.array([integrand_za_approx(z0, z, wc, l, n, t).imag for z in z_arr])

# plotting the error in the approximation of the imaginary part 

diff = np.array([mp.fabs(d) for d in (za_arr_1_imag - za_arr_2_imag)])

plt.plot(z_arr, -za_arr_1_imag, 'o', markersize = 2)
plt.plot(z_arr, -za_arr_2_imag, 'o', markersize = 2)
plt.plot(z_arr, -diff / za_arr_1_imag)
plt.yscale('log')
plt.axvline(z0*0.9999, linestyle='dashed')
plt.axvline(z0*1.0001, linestyle='dashed')
plt.axhline(0.01, linestyle='dashed')

# plotting the error in the approximation of the real part 

diff = np.array([mp.fabs(d)  for d in (za_arr_1_re - za_arr_2_re)])
diff /= za_arr_1_re
plt.plot(z_arr, za_arr_1_re, 'o', markersize = 2)
plt.plot(z_arr, za_arr_2_re, 'o', markersize = 2)
plt.plot(z_arr, diff)
plt.axvline(z0*0.9999, linestyle='dashed')
plt.axvline(z0*1.0001, linestyle='dashed')
plt.axhline(0.01, linestyle='dashed')
plt.yscale('log')
