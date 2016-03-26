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

# approx Re(el) by second order Taylor expansion
def integral_za_approx(z1, z2, z0, wc, l, n, t):
    el = BiMax.d_l(z0, wc, n, t)
    el_img = el.imag
    dz_el_re = BiMax.dz_dl(z0, wc, n, t).real
    d2z_el_re = BiMax.dz2_dl(z0, wc, n, t).real
    
    def helper(z):
        kl = wc*l/mp.sqrt(2)/z
        ka = kl * al_ratio
#         if ka < 0.4:
#             j02 = 1 - 0.5 * ka**2
        num = f1(kl) * j0(ka)**2
        denom = z**2 * mp.mpc(dz_el_re * (z-z0) + 0.5 * d2z_el_re * (z-z0)**2, el_img)
        return num / denom
    return mp.quad(helper, [z1, z0, z2])

wrel, l, n, t, tc = 1.01, 5, 0.1, 3, 1
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

mp.mp.dps = 25
guess = z_b(wc, n, t)
z0 = mp.findroot(lambda z: BiMax.d_l(z, wc, n, t).real, guess)

z1 = z0* 0.9999
z2 = z0*1.0001

limits = [z1, z2]
limits_full = [z1, z0, z2]
#int0 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits)
int1 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits_full)
int2 = integral_za_approx(z1, z2, z0, wc, l, n, t)
#int3 = integral_za_approx_2(z1, z2, z0, wc, l, n, t)

#print('int0 is ', int0)
print('int1 is ', int1)
print('int2 is ', int2)
#print('int3 is ', int3)
mp.mp.dps = 15

z_arr = z0 * np.linspace(0.999, 1.001, 200)

za_arr_1_re = np.array([integrand_za(z, wc, l, n, t).real for z in z_arr])
za_arr_1_imag = np.array([integrand_za(z, wc, l, n, t).imag for z in z_arr])
za_arr_2_re = np.array([integrand_za_approx(z0, z, wc, l, n, t).real for z in z\
_arr])
za_arr_2_imag = np.array([integrand_za_approx(z0, z, wc, l, n, t).imag for z in\
 z_arr])

# plotting the error in the approximation of the imaginary part

# diff = np.array([mp.fabs(d) for d in (za_arr_1_imag - za_arr_2_imag)])

# plt.plot(z_arr, -za_arr_1_imag, 'o', markersize = 2)
# plt.plot(z_arr, -za_arr_2_imag, 'o', markersize = 2)
# plt.plot(z_arr, -diff / za_arr_1_imag)

diff = np.array([mp.fabs(d) for d in (za_arr_1_re - za_arr_2_re)])

#plt.plot(z_arr, za_arr_1_re, 'o', markersize = 2)
plt.plot(z_arr, za_arr_2_re, 'o', markersize = 2)
plt.plot(z_arr, diff / za_arr_1_re)

plt.yscale('log')
plt.axvline(z0*0.9999, linestyle='dashed')
plt.axvline(z0*1.0001, linestyle='dashed')
plt.axhline(0.01, linestyle='dashed')

