import numpy as np
import sys
import os
from bit_flip_sweep import sweep_param, Dev
from FQ_sympy_functions import RoundDevice

sys.path.append(os.path.expanduser('~/source'))
import kyle_tools as kt

trial_N = 10_000
test_N = 50_000

L_range = Dev.L * np.linspace(.5,1,2)
g_range = np.linspace(8,16,3)

#Dev.change_vals({'dbeta':.1})
dt = 1/100

RDev = RoundDevice


g_dict = {'param':'gamma', 'sweep':g_range}
L_dict = {'param':'L', 'sweep': L_range}

def save_name(sweep_dict, Dev):
    time = sweep_dict['start_time']
    return f"L{10E11*Dev.L:.0f}_{time.hour:02d}{time.minute:02d}_{time.second:02d}"

def double_sweep(dict_1=L_dict, dict_2=g_dict, Dev=Dev, **kwargs):
    param = dict_1['param']
    param_vals = dict_1['sweep']
    for i, param_val in enumerate(param_vals):

        print('{}={}; {} of {}'.format(param, param_val, i+1, len(param_vals)))

        Dev.change_vals({param:param_val})

        d = run_sweep(sweep_dict=dict_2, Dev=Dev, **kwargs);
        del(d)


def run_sweep(sweep_dict=L_dict, Dev=Dev, N_t=trial_N, timestep=dt, save_dir=None, naming_func=save_name, **kwargs):

    param_sweep = sweep_param(Dev=Dev, sweep_dict=sweep_dict, N=N_t, N_test=test_N, delta_t=timestep, **kwargs);
    if naming_func is not None:
        param_sweep['save_name'] = naming_func(param_sweep, Dev)
    
    kt.save_as_json(param_sweep, dir=save_dir)

    return param_sweep


