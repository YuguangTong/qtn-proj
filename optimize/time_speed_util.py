from timeit import timeit, repeat

def time_f1(x):
    """
    time f1 function
    """
    num = 1000
    return timeit('f1({0})'.format(x),
                  setup = 'from qtn.util import f1;',
                  number = num) / num
def time_f1_sp(x):
    """
    time f1 function
    """
    num = 1000
    return timeit('f1_sp({0})'.format(x),
                  setup = 'from qtn.bimax_util import f1_sp;',
                  number = num) / num


def time_d_l(z, wc, n, t):
    """
    time d_l function
    """
    setup_str = 'from qtn.bimax import BiMax;' +\
        'ant_len, ant_rad, base_cap  = 50, 1.9e-4, 20e-12;' + \
        'p = BiMax(ant_len, ant_rad, base_cap);'
    time_str = 'p.d_l({0}, {1}, {2}, {3})'.format(z, wc, n, t)
    num = 100
    return timeit(time_str, setup = setup_str, number = num) / num

def time_d_l_sp(z, wc, n, t):
    """
    time d_l_sp function
    """
#     setup_str = 'from qtn.bimax import BiMax;' +\
#         'ant_len, ant_rad, base_cap  = 50, 1.9e-4, 20e-12;' + \
#         'p = BiMax(ant_len, ant_rad, base_cap);'
#     time_str = 'p.d_l({0}, {1}, {2}, {3})'.format(z, wc, n, t)
    setup_str = 'from qtn.bimax_util import d_l_sp;'
    time_str = 'd_l_sp({0}, {1}, {2}, {3})'.format(z, wc, n, t)
    num = 100
    return timeit(time_str, setup = setup_str, number = num) / num

def time_long_interval(w, n, t):
    """
    time long_interval function
    """
    setup_str = 'from qtn.bimax import BiMax;' +\
        'ant_len, ant_rad, base_cap  = 50, 1.9e-4, 20e-12;' + \
        'p = BiMax(ant_len, ant_rad, base_cap);'
    time_str = 'p.long_interval({0},{1}, {2})'.format(w, n, t)
    num = 20
    return timeit(time_str, setup = setup_str, number = num) / num

def time_long_interval_sp(w, n, t):
    """
    time long_interval function
    """
#     setup_str = 'from qtn.bimax import BiMax;' +\
#         'ant_len, ant_rad, base_cap  = 50, 1.9e-4, 20e-12;' + \
#         'p = BiMax(ant_len, ant_rad, base_cap);'
#     time_str = 'p.long_interval({0},{1}, {2})'.format(w, n, t)
    setup_str = 'from qtn.bimax_util import long_interval_sp;'
    time_str = 'long_interval_sp({0},{1}, {2})'.format(w, n, t)
    num = 20
    return timeit(time_str, setup = setup_str, number = num) / num

def time_za_l_integrand(z, wc, l, n, t):
    """
    time za_l_integrand
    """
    setup_str = 'from qtn.bimax import BiMax;' +\
        'ant_len, ant_rad, base_cap  = 50, 1.9e-4, 20e-12;' + \
        'p = BiMax(ant_len, ant_rad, base_cap);'
    time_str = 'p.za_l_integrand({0},{1}, {2}, {3}, {4})'.\
        format(z, wc, l, n, t)
    num = 10
    return timeit(time_str, setup = setup_str, number = num) / num

def time_za_l(wc, l, n, t, tc):
    """
    time za_l
    """
    setup_str = 'from qtn.bimax import BiMax;' +\
        'ant_len, ant_rad, base_cap  = 50, 1.9e-4, 20e-12;' + \
        'p = BiMax(ant_len, ant_rad, base_cap);'
    time_str = 'p.za_l({0},{1}, {2}, {3}, {4})'.\
        format(wc, l, n, t, tc)
    num = 10
    return timeit(time_str, setup = setup_str, number = num) / num

print('f1 takes {0:.3g}ms'.format(1000 * time_f1(3.3)))
print('f1_sp takes {0:.3g}ms'.format(1000 * time_f1_sp(3.3)))
print('d_l takes {0:.3g}ms'.format(1000*time_d_l(0.1, 1.2, 0.05, 10)))
print('d_l_sp takes {0:.3g}ms'.format(1000*time_d_l_sp(0.1, 1.2, 0.05, 10)))
print('long_interval takes {0:.3g}s'.format(time_long_interval(1.02, 0, 1)))
print('long_interval_sp takes {0:.3g}s'.format(time_long_interval_sp(1.02, 0, 1)))
print('za_l_integrand takes {0:.3g}ms'.
      format(1000 * time_za_l_integrand(10., 0.1, 5.3, 0.1, 8.)))
print('za_l takes {0:.3g}s'.format(time_za_l(0.1, 5.3, 0.1, 8., 1.6e5)))
