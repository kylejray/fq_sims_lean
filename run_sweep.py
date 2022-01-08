import numpy as np
from bit_flip_sweep import sweep_param, Dev
from FQ_sympy_functions import RoundDevice

trial_N = 10_000
test_N = 50_000
L_range = Dev.L * np.linspace(.5,1,2)
g_range = np.linspace(8,16,5)
#Dev.change_vals({'dbeta':.1})
dt = 1/100

RDev = RoundDevice


g_dict = {'param':'gamma', 'sweep':g_range}
L_dict = {'param':'L', 'sweep': L_range}

def set_ell_dict(L):
    return {'param':'ell', 'sweep':L/(2*np.linspace(4, 20, 8))}

def double_sweep(dict_1=L_dict, dict_2=g_dict, Dev=Dev, **kwargs):
    param = dict_1['param']
    param_vals = dict_1['sweep']
    for i, param_val in enumerate(param_vals):

        print('{}={}; {} of {}'.format(param, param_val, i+1, len(param_vals)))

        Dev.change_vals({param:param_val})

        d = run_sweep(sweep_dict=dict_2, Dev=Dev, **kwargs);
        del(d)


def run_sweep(sweep_dict=L_dict, Dev=Dev, N_t=trial_N, timestep=dt, **kwargs):
    return sweep_param(Dev=Dev, sweep_dict=sweep_dict, N=N_t, N_test=test_N, delta_t=timestep, **kwargs);


