import numpy as np
from bit_flip_sweep import sweep_param, Dev

trial_N = 10_000
test_N = 10_000
L_range = Dev.L * np.linspace(.25,.85,5)
ell_range = Dev.ell * np.linspace(.5,1.5,5)
I0 = Dev.I_minus
Dev.change_vals({'I_minus':0*I0/5})
dt = 1/100

ell_dict = {'param':'ell', 'sweep':ell_range}
L_dict = {'param':'L', 'sweep': L_range}

def set_ell_dict(L):
    return {'param':'ell', 'sweep':L/(2*np.linspace(4, 20, 8))}

def run_double_sweep(dict_1=L_dict, dict_2=ell_dict, dict_2_function=set_ell_dict, **kwargs):
    param = dict_1['param']
    param_vals = dict_1['sweep']
    for i, param_val in enumerate(param_vals):

        print('{} {} of {}'.format(param, i+1, len(param_vals)))
        Dev.change_vals({param:param_val})

        if dict_2_function is not None:
            dict_2 = dict_2_function(param_val)

        sweep_param(sweep_dict=dict_2,  minimize_ell=False, **kwargs)


def run_sweep(sweep_dict=L_dict, N_t=trial_N, timestep=dt, **kwargs):
    return sweep_param(Dev=Dev, sweep_dict=sweep_dict, N=N_t, N_test=test_N, delta_t=timestep, **kwargs)


