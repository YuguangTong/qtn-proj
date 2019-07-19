from qtn.bimax import BiMax

import numpy as np
from sympy.mpmath import mp,fp
import matplotlib.pyplot as plt
%matplotlib inline

from qtn.new_bimax import new_BiMax

ant_len = 50      # m (monopole) 
ant_rad = 1.9e-4  # m
base_cap = 20e-12 # Farad
al_ratio = ant_rad / ant_len
print(al_ratio)

# instantiate a copy of the old and the new object
q1 = BiMax(ant_len, ant_rad, base_cap)
q2 = new_BiMax(ant_len, ant_rad, base_cap)

wrel, l, n, t, tc = 0.8, 5, 0.2, 10, 1

mp.dps = 15
print(q2.new_bimax(wrel, l, n, t, tc))
mp.dps= 15

mp.dps = 15
print(q1.bimax(wrel, l, n, t, tc))
mp.dps= 15

# calculate a spectrum

wrel_arr = np.array([0.1, 0.5, 0.6, 0.8, 0.9, 0.95, 1.01, 1.03, \
                         1.05, 1.08, 1.1, 1.2, 1.3, 1.5, \
                         1.7, 2.0, 2.3, 2.6, 3.0, 4, 6])

v2e_arr = np.array([q2.new_bimax(wrel, l, n, t, tc) for wrel in wrel_arr])

plt.plot(wrel_arr, v2e_arr, '-or', markersize = 1)
plt.axvline(1.0, linestyle = 'dashed')
plt.yscale('log')
plt.xscale('log')
plt.axhline(v2e_arr[0], linestyle = 'dashed')



