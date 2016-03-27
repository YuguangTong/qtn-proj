# verify that the peak of the za integrand can be neglected

from qtn.bimax import BiMax
from qtn.util import f1, j0
from qtn.bimax_util import z_b
import numpy as np
from sympy.mpmath import mp,fp
import matplotlib.pyplot as plt
%matplotlib inline

ant_len = 50      # m (monopole) 
ant_rad = 1.9e-4  # m
base_cap = 20e-12 # Farad
al_ratio = ant_rad / ant_len
print(al_ratio)

# integrand of Za
def integrand_za(z, wc, l, n, t):
    kl = wc*l/fp.sqrt(2)/z
    ka = kl * al_ratio
    num = f1(kl) * j0(ka)**2
    denom = z**2 * BiMax.d_l(z, wc, n, t)
    return num/denom

wrel, l, n, t, tc = 1.01, 5, 0, 10, 1
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



mp.dps = 15
#guess = z_b(wc, n, t)
#z0 = fp.findroot(lambda z: BiMax.d_l(z, wc, n, t).real, guess)

z1 = z0 * 0.999
z2 = z0 * 1.001
limits_1 = [z1, z0, z2]
limits_2 = [0.1, z1]
limits_3 = [z2, mp.inf]
limits_4 = [0.1, z0, mp.inf]
int_1 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits_1)
int_2 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits_2)
int_3 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits_3)
int_4 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits_4)


# print(int_1)
# print(int_2)
# print(int_3)
print(int_4, '\n')

print('approx tot = ', int_2.real + int_3.real)
tot = int_1 + int_2 + int_3
print('error = ', fp.fabs(int_1.real) / tot.real)

mp.dps = 35
guess = z_b(wc, n, t)
z0 = mp.findroot(lambda z: BiMax.d_l(z, wc, n, t).real, guess)

z1 = z0 * 0.999
z2 = z0 * 1.001
limits_1 = [z1, z0, z2]
limits_2 = [0.1, z1]
limits_3 = [z2, mp.inf]
limits_4 = [0.1, z0, mp.inf]
# int_1 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits_1)
# int_2 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits_2)
# int_3 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits_3)
int_4 = mp.quad(lambda z: integrand_za(z, wc, l, n, t), limits_4)

# print(int_1)
# print(int_2)
# print(int_3)
print(int_4, '\n')

# print('approx tot = ', int_2.real + int_3.real)
# tot = int_1 + int_2 + int_3
# print('error = ', fp.fabs(int_1.real) / tot.real)
mp.dps = 15
