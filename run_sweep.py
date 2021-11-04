import numpy as np
from bit_flip_sweep import sweep_L, Dev

trial_N = 10_000
test_N = 50_000
L_range = Dev.L * np.linspace(.5,1.5,5)
ell_range = Dev.ell * np.linspace(.5,1.5,5)
I0 = Dev.I_minus
Dev.change_vals({'I_minus':0*I0/10})
dt = 1/200

def run_ell_sweep(ell=ell_range, **kwargs):
    
    for i,item in enumerate(ell):
        print('ell {} of {}'.format(i+1, len(ell)))
        Dev.change_vals({'ell':item})
        
        run_L_sim(**kwargs, minimize_ell=False)


def run_L_sim(L=L_range, N_t=trial_N, timestep=dt, **kwargs):
    return sweep_L(Dev=Dev, L_vals=L, N=N_t, N_test=test_N, delta_t=timestep, **kwargs)













