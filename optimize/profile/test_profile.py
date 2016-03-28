from qtn.new_bimax import new_BiMax
from mpmath import fp

ant_len = 45 #50      # m (monopole) 
ant_rad = 2e-4 #1.9e-4  # m
base_cap = 45e-12 # 20e-12 # Farad

p2 = new_BiMax(ant_len, ant_rad, base_cap)

wrel, l, n, t, tc = 0.2, 2, 0, 5, 1
wc = wrel * fp.sqrt(1 + n)

print(p2.gamma(wrel, l, n, t, tc))
