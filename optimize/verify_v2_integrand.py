from qtn.bimax import BiMax
from qtn.util import f1
from qtn.bimax_util import z_b
import numpy as np
import sympy.mpmath as mp
import matplotlib.pyplot as plt
%matplotlib inline

ant_len = 50      # m (monopole) 
ant_rad = 1.9e-4  # m
base_cap = 20e-12 # Farad

p = BiMax(ant_len, ant_rad, base_cap)

wrel, l, n, t, tc = 1.03, 5, 0.1, 10, 1
wc = wrel * np.sqrt(1 + n)

#p.bimax_sharp_peak(wrel, l, n, t, tc)
guess = z_b(wc, n, t)
print(guess)
z0 = mp.findroot(lambda z: BiMax.d_l(z, wc, n, t).real, guess)
print(z0)

BiMax.d_l(z0, wc, n, t)

# integrand of V^2

def integrand(z, wc, l, n, t):
    return f1(wc*l/z/mp.sqrt(2)) * z * \
            (mp.exp(-z**2) + n/mp.sqrt(t)*mp.exp(-z**2 / t)) / \
            mp.fabs(BiMax.d_l(z, wc, n, t))**2 / wc**2


def integrand_approx(z0, z, wc, n, t):
    return f1(wc*l/z/mp.sqrt(2)) * z * \
            (mp.exp(-z**2) + n/mp.sqrt(t)*mp.exp(-z**2 / t)) / \
            (mp.fabs(BiMax.d_l(z0, wc, n, t).imag)**2 + \
             BiMax.dz_dl(z0, wc, n, t).real**2 * (z - z0)**2) / wc**2
        
def integrand_approx_2(z0, z, wc, n, t):
    return f1(wc*l/z0/mp.sqrt(2)) * z0 * \
            (mp.exp(-z0**2) + n/mp.sqrt(t)*mp.exp(-z0**2 / t)) / \
            (mp.fabs(BiMax.d_l(z0, wc, n, t).imag)**2 + \
             BiMax.dz_dl(z0, wc, n, t).real**2 * (z - z0)**2) / wc**2

# compare 1/|dl|^2 and 1/(dl.imag **2 + dz_dl.real**2 (z-z0)**2)

z_arr = z0 * np.linspace(0.999, 1.001, 100)
#denom_1 = np.array([1/ mp.fabs(BiMax.d_l(z, wc, n, t))**2 for z in z_arr])
denom_1 = np.array([integrand(z, wc, l, n, t) for z in z_arr])

# denom_2 = np.array([1 / (mp.fabs(BiMax.d_l(z0, wc, n, t).imag)**2 + 
#              BiMax.dz_dl(z0, wc, n, t).real**2 * (z - z0)**2) for z in z_arr ])

denom_2 = np.array([integrand_approx(z0, z, wc, n, t) for z in z_arr ])

diff = np.array([ mp.fabs(d) for d in (denom_1 - denom_2)])
#z_intgrd = [p.bimax_smooth_integrand(z, wrel, l, n, t, tc) for z in z_arr]

# plot to show how good the approximation is

# texts in the plot
texts = "{0}{1:.3g}\n".format(r'$\omega/\omega_{pT} = $', wrel) + \
    "{0}{1:.3g}\n".format(r'$l/l_D = $',l) + \
    "{0}{1:.3g}\n".format(r'$n_h/n_c = $',n) + \
    "{0}{1:.3g}\n".format(r'$T_h/T_c = $',t) +\
    "v dashed: $\pm10^{-4}\zeta_0$ \n" + \
    "h dashed: $\pm10^{-4}\zeta_0$" 
fig = plt.figure(figsize = [6, 4])
plt.plot(z_arr, denom_1, 'r', label = 'exact')
plt.plot(z_arr, denom_2, 'b', label = 'approximate')
plt.plot(z_arr, diff, 'g', label = 'difference')
plt.plot(z_arr, diff/denom_1, label = 'relative diff')
plt.axhline(0.01, linestyle='dashed')
plt.axvline(z0*0.9999, linestyle='dashed')
plt.axvline(z0*1.0001, linestyle='dashed')
plt.yscale('log')
plt.legend(loc = 'right', frameon = False)
plt.xlabel(r'$\zeta$')
plt.ylabel(r'$V^2$ integrand')
plt.annotate(texts, xy = (0.05, 0.4), xycoords='axes fraction')
plt.show()

fig_dir = '/Users/ygtong/Google_Drive/research/projects/QTN/Tong/log/qtn_speedup/figures'
fig_name = fig_dir + '/v2integrand_bimax_4.png'
#fig.savefig(fig_name, dpi = 100)
