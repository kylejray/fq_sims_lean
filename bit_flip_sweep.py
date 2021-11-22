import sys
import os
import numpy as np 
import datetime
import json
import copy
from simtools.infoenginessims.simprocedures.basic_simprocedures import ReturnFinalState

sys.path.append(os.path.expanduser('~/source'))

from quick_sim import setup_sim
from kyle_tools import separate_by_state, jsonify
from sus.protocol_designer import System
from sus.library.fq_systems import fq_pot


sys.path.append(os.path.expanduser('~/source/simtools/'))
from infoenginessims.api import *
from infoenginessims.simprocedures import basic_simprocedures as sp
from infoenginessims.simprocedures import running_measurements as rp
from infoenginessims.simprocedures import trajectory_measurements as tp

from FQ_sympy_functions import find_px, DeviceParams, fidelity



#params 1:
kT = .41*1.38E-23
Dev = DeviceParams()

'''
#params 2:
kT = 6.9E-24
value_dict = {'C':530E-15, 'R':2.1, 'L':140E-12}
Dev2 = DeviceParams(value_dict)
'''

#these are some relevant dimensionful scales: Dev.alpha is the natural units for the JJ fluxes and U_0 is the natural scale for the potential
#IMPORTANT: all energies are measured in units of Dev.U_0
#default_real = (.084, -2.5, 12, 6.2, .2)


#these are important dimensionless simulation quantities, accounting for 
#m being measured in units of C, lambda in units of 1/R, energy in units of U_0

m_prime = np.array((1, 1/4))
lambda_prime = np.array((2, 1/2))
kT_prime = kT/Dev.U_0

L_sweep = Dev.L*(np.linspace(.1,1,7))

L_dict={'param':'L', 'sweep':L_sweep}

def sweep_param(Dev=Dev, sweep_dict=L_dict, N=10_000, N_test=2500, delta_t=1/200, save_dir='./', minimize_ell=False, kT_prime=kT_prime):
    param_vals = sweep_dict['sweep']
    param = sweep_dict['param']
    cnt=0
    date = datetime.datetime.now().strftime('%d_%m_%Y_%H%M_%S')
    output_dict={'kT_prime':kT_prime, 'date':date}
    output_dict['param_sweep'] = param_vals
    output_dict['sim_results'] = []

    for param_val in param_vals:
        cnt += 1
        print("\n {} {} of {}".format(param, cnt, len(param_vals)))
        temp_dict = {}

        Dev.change_vals({param:param_val})

        try: store_sys, comp_sys = set_systems(Dev, comp_tau=10)
        except AssertionError: continue

        init_state = generate_eq_state(store_sys, N_test, kT_prime)
        try: verify_eq_state(init_state)
        except AssertionError:
            print('\n bad initial_state at param change check')
            temp_dict['store_params'] = store_sys.protocol.params
            temp_dict['comp_params'] = comp_sys.protocol.params
            temp_dict['device'] = copy.deepcopy(Dev).__dict__
            temp_dict['initial_state'] = init_state
            temp_dict['terminated'] = True
            output_dict['sim_results'].append(temp_dict)
            continue
        

        prelim_sim = generate_sim(comp_sys, init_state, Dev, 2*delta_t, set_mean_procs)
        prelim_sim.output = prelim_sim.run()

        z_states = prelim_sim.output.zero_means['values']
        o_states = prelim_sim.output.one_means['values']
        times = np.linspace(0, comp_sys.protocol.t_f, prelim_sim.nsteps+1)
        

        prelim_d={}
        prelim_d = prelim_sim.output.__dict__
        prelim_d['dt'] = prelim_sim.dt
        prelim_d['nsteps'] = prelim_sim.nsteps
        prelim_d['store_params'] = store_sys.protocol.params
        prelim_d['comp_params'] = comp_sys.protocol.params
        prelim_d['device'] = copy.deepcopy(Dev).__dict__
        temp_dict['quantum'] = 1.054E-34 / (np.sqrt(Dev.L*Dev.C)*kT_prime*Dev.U_0)
        temp_dict['prelim_sim'] = prelim_d


        try: tau_candidates, t_crit = get_tau_candidate(z_states, o_states, times)
        except AssertionError:
            temp_dict['terminated'] = True
            continue
        prelim_d['tau_list'] = tau_candidates
        prelim_d['t_crit'] = t_crit

        

        if minimize_ell:
            sys_candidates = []
            test_times=[]
            t_crit=[]
            Devs=[]
            ell_sims = []
            for idx, item in enumerate(tau_candidates):
                temp_Dev = copy.deepcopy(Dev)
                temp_store, temp_comp = store_sys, comp_sys
                t_p, t_pdc = t_crit, item
                print('\n', 'one side initial_vals', t_p, t_pdc, t_p/t_pdc, temp_Dev.gamma)
                old_ratio = 3
                i=0
                while abs(t_p-t_pdc-1) > .025 and i<10:
                    ell_d = {}
                
                    old_ratio, i_plus = change_ell(temp_Dev, t_p, t_pdc, old_ratio)
                    i += i_plus

                    try: temp_store, temp_comp = set_systems(temp_Dev, comp_tau=10)
                    except AssertionError: continue

                    '''
                    for item in [temp_store, temp_comp]:
                        item.protocol.params[2,:] = temp_Dev.gamma
                    '''
                    for item in [temp_store,temp_comp]:
                        print(item.protocol.params[:,0])

                    init_state = generate_eq_state(store_sys, N_test, kT_prime)

                    try: verify_eq_state(init_state, verbose=True)
                    except AssertionError:
                        print('\n ran into bad initial state in ell sweep and cancelled it')
                        i=10
                        continue

                    ell_sim = generate_sim(comp_sys, init_state, temp_Dev, 2*delta_t, set_mean_procs)
                    ell_sim.output = prelim_sim.run()

                    z_states = ell_sim.output.zero_means['values']
                    o_states = ell_sim.output.one_means['values']

                    t_pdc, t_p = get_tau_candidate(z_states, o_states, times)
                    t_pdc=t_pdc[idx]

                    print('\n', t_p, t_pdc, t_p/t_pdc, temp_Dev.gamma)

                    ell_d = ell_sim.output.__dict__
                    ell_d['gamma'] = copy.deepcopy(temp_Dev).gamma
                    ell_d['t_p'] = t_p
                    ell_d['t_pdc'] = t_pdc

                    ell_sims.append(ell_d)
                
                test_times[idx] = t_pdc
                t_crit[idx] = t_p
                sys_candidates.append([temp_store, temp_comp])
                Devs.append(temp_Dev)


        else:
            Devs = [Dev]
            t_crit = [t_crit]
            test_times= [tau_candidates]
            sys_candidates= [[store_sys, comp_sys]]
        '''
        if minimize_ell:
            temp_dict['ell_sims'] = ell_sims
        '''
        temp_dict['tau_list']=[]
        temp_dict['sims'] = []

        for curr_sys, tau_list, t_crit, Dev in list(zip(sys_candidates, test_times, t_crit, Devs)):

            store_sys, comp_sys = curr_sys
            init_state = generate_eq_state(store_sys, N, kT_prime)
            try: verify_eq_state(init_state)
            except AssertionError:
                print('\n unexpected bad initial_state')
                continue
            
            sweep_tau(Dev, t_crit, tau_list, init_state, comp_sys, store_sys, delta_t, temp_dict)

        min_work, w_idx = get_best_work(temp_dict['sims'])


        temp_dict['min_work'] = min_work
        temp_dict['min_work_index'] = w_idx
        output_dict['sim_results'].append(temp_dict)
    
    if save_dir is not None:
        output_dict['save_name'] = ''
        save_sweep(output_dict, dir=save_dir)
    
    return output_dict


def save_sweep(output_dict, dir='./'):
    if not os.path.exists(dir):
        os.makedirs(dir)
    save_dict = jsonify(output_dict)
    name = output_dict['save_name']
    dir += name + output_dict['date'] + '.json'
    with open(dir, 'w') as fout:
        json.dump(save_dict, fout)
    print('\n saved as json')
    return

def set_systems(Device, eq_tau=1, comp_tau=1, d_store_comp=[.2,.2]):

    assert 4*Device.gamma > Device.beta, '\n gamma must be >beta/4, it is set too small'
    assert Device.beta > 1, '\n beta < 1, it is set too small'

    g, beta, dbeta = Device.gamma, Device.beta, Device.dbeta
    pxdc_crit = -2*np.arccos(1/beta)+ (beta/(2*g))*np.sqrt(1-1/beta**2)
    pxdc_store, pxdc_comp = pxdc_crit+d_store_comp[0], pxdc_crit-d_store_comp[1]

    px_store = find_px(dbeta, pxdc_store, mode='min_of_max')
    px_comp = find_px(dbeta, pxdc_comp, mode='min_of_mid')

    s_params = [px_store, pxdc_store, g, beta, dbeta]
    c_params = [px_comp, pxdc_comp, g, beta, dbeta]

    fq_pot.default_params = s_params
    store_sys = System(fq_pot.trivial_protocol(), fq_pot)
    store_sys.protocol.t_f = eq_tau
    store_sys.mass= m_prime

    fq_pot.default_params = c_params
    comp_sys = System(fq_pot.trivial_protocol(), fq_pot)
    comp_sys.protocol.t_f = comp_tau
    comp_sys.mass = m_prime

    return(store_sys, comp_sys)

def generate_eq_state(eq_sys, N, kT_prime, domain=None):
    pxdc_store = eq_sys.protocol.params[1,0]
    gamma_store = eq_sys.protocol.params[2,0]
    if domain is None:
        range = np.e/np.sqrt(gamma_store/2)
        domain = [[-4, pxdc_store-2*range], [4, pxdc_store+2*range]]
    
    init_state = eq_sys.eq_state(N, beta=1/(kT_prime), manual_domain=domain, axes=[1,2], verbose=False)
    return init_state

def info_state_means(initial_state, info_subspace=np.s_[...,0,0]):
    info_state_means=[]
    is_bools = separate_by_state(initial_state[info_subspace])
    for key in is_bools.keys():
        info_state_means.append(initial_state[is_bools[key]].mean(axis=0))
    return np.array(info_state_means)

def generate_sim(comp_sys, initial_state, Device, delta_t, proc_function, **proc_kwargs):
    gamma = (lambda_prime/m_prime) * np.sqrt(Dev.L*Device.C) / (Device.R*Device.C) 
    theta = 1/m_prime
    eta = (Device.L/(Device.R**2 * Device.C))**(1/4) * np.sqrt(kT_prime*lambda_prime) / m_prime
    
    procs = proc_function(initial_state, **proc_kwargs)

    return setup_sim(comp_sys, initial_state, procedures=procs, 
                sim_params=[gamma, theta, eta], dt=delta_t)

def set_mean_procs(initial_state):
    is_bools = separate_by_state(initial_state[...,0,0])
    mean_procs = [
            sp.ReturnInitialState(),
            sp.ReturnFinalState(),
            sp.MeasureMeanValue(rp.get_current_state, output_name = 'zero_means', trial_request=is_bools['0']),
            sp.MeasureMeanValue(rp.get_current_state, output_name = 'one_means', trial_request=is_bools['1'])
            ]
    return mean_procs

def set_mean_evolution_procs(info_state_means):
    mean_evo_procs = [
        sp.ReturnFinalState(),
        sp.ReturnInitialState(),
        sp.MeasureStepValue(rp.get_current_state, trial_request=np.s_[0], output_name='zero_means'),
        sp.MeasureStepValue(rp.get_current_state, trial_request=np.s_[1], output_name='one_means')
    ]
    return mean_evo_procs

def set_bundle_evo_procs(state_bundle, weights=None):
    is_bools = separate_by_state(state_bundle[...,0,0])
    w_z = weights[is_bools['0']]
    w_o = weights[is_bools['1']]

    bundle_evo_procs = [
        sp.ReturnFinalState(),
        sp.ReturnInitialState(),
        sp.MeasureAllState(trial_request=np.s_[:1000]),
        sp.MeasureMeanValue(rp.get_current_state, trial_request=is_bools['0'], output_name='zero_means', weights=w_z),
        sp.MeasureMeanValue(rp.get_current_state, trial_request=is_bools['1'], output_name='one_means', weights=w_o)
    ]
    return bundle_evo_procs

def set_general_procs(initial_state, all_state_skip=5):
    is_bools = separate_by_state(initial_state[...,0,0])
    real_procedures = [
              sp.ReturnFinalState(),
              sp.ReturnInitialState(),
              sp.MeasureAllState(trial_request=slice(0, 100), step_request=np.s_[::all_state_skip]),  
              sp.MeasureMeanValue(rp.get_kinetic, output_name='kinetic' ),
              sp.MeasureMeanValue(rp.get_potential, output_name='potential'),
              sp.MeasureMeanValue(rp.get_current_state, output_name = 'zero_means', trial_request=is_bools['0']),
              sp.MeasureMeanValue(rp.get_current_state, output_name = 'one_means',  trial_request=is_bools['1']),
              tp.CountJumps(state_slice=np.s_[...,0,0])
             ]
    return real_procedures

def find_zero(a, t, burn_in=0, mode='decreasing'):
    D = -2
    if mode is 'increasing':
        D = 2
    assert len(a)==len(t), '\n a and t need to be same length'
    for i,item in enumerate(a):
        if i > burn_in:
            if np.sign(item) - np.sign(last_item) == D:
                t_i = t[i-1] + (t[i]-t[i-1])* last_item/(last_item+abs(item))
                return i, t_i
        last_item = item

def get_tau_candidate(z_means, o_means, t):
    t_list =[[],[]]
    burn = int(1/(t[1]-t[0]))
    i_z, t_z = find_zero(z_means[...,0,1], t, burn_in=burn)
    i_o, t_o = find_zero(o_means[...,0,1], t, burn_in=burn, mode='increasing')
    i_crit = int((i_z+i_o)/2)
    t_crit = (t_o + t_z)/2

    assert np.sign(z_means[i_crit,0,0]) == 1 and np.sign(o_means[i_crit,0,0]) == -1, 'not a good swap'

    for item in [z_means, o_means]:
        t_list[1].append(find_zero(item[i_crit:,1,1], t[i_crit:])[1])
        dt_left = find_zero(item[i_crit::-1,1,1],-t[i_crit::-1]+t[i_crit], mode='increasing')[1]
        t_list[0].append(t[i_crit]-dt_left)

    for i,item in enumerate(t_list):
        t_list[i] = np.mean(item)
    
    return t_list, t_crit

def verify_eq_state(init_state, check_symmetry=False, verbose=False):
    phi = init_state[...,0,0]
    inf_states = separate_by_state(phi)
    n_z, n_o = sum(inf_states['0']), sum(inf_states['1'])

    assert  .4 < n_z/n_o < 2.5

    if check_symmetry:
        
        symmetry = False
        if n_z/n_o > .9 and n_z/n_o < 1.1: symmetry=True
        assert symmetry is True

    phi_z, phi_o = phi[inf_states['0']], phi[inf_states['1']]
    max_z = np.mean(phi_z)+ 3*np.std(phi_z)
    min_o = np.mean(phi_o)- 3*np.std(phi_o)
    separation=False
    if min_o > max_z:
        separation = True
    assert separation is True

    if verbose:
        if n_z/n_o <.9 or n_z/n_o > 1.1:
            print('n_z:{},n_o:{}'.format(n_z,n_o))
        if min_o-max_z < 1:
            print('max_z: {}, min_o: {}'.format(max_z, min_o))

def change_ell(Device, t_p, t_pdc, previous_ratio, test_mode=False):
    current_ell = Device.ell
        
    if abs(t_p/t_pdc - 1) <= .95*abs(previous_ratio-1):

        #new_ell = current_ell * ((t_p/t_pdc)**2 +1)/2
        new_ell = current_ell * (t_p/t_pdc)**2
        new_ratio = t_p/t_pdc
        i_plus = 1
    else:
        #new_ell = 2*current_ell /(previous_ratio**2+1)
        new_ell = current_ell * (1/previous_ratio**2)
        new_ratio = previous_ratio
        i_plus = 10
    
    if test_mode:
        new_ell = current_ell + np.sign(t_p/t_pdc-1)*current_ell * .2
        new_ratio = t_p/t_pdc
        i_plus=1

    Device.change_vals({'ell':new_ell})
    return new_ratio, i_plus

def newtime(t1, w1, times, works, iter, delta_t):
    if iter==15:
        return t1, [t1, w1] , iter
    t2 = times[iter]
    w2 = works[iter]
    if np.isclose(w1, w2, atol=.0005):
        return (t1+t2)/2, [t1, w1], 15
    if w2 < w1:
        return t2+delta_t*np.sign(t2-t1), [t2, w2], iter+1
    if w2 > w1:
        return t1-delta_t*np.sign(t2-t1), [t1, w1], iter+1
        

def sweep_tau(Dev, t_crit, tau_list, init_state, comp_sys, store_sys, delta_t, write_dict, tau_resolution=.025):
    tau_cnt = 0

    for tau in tau_list:
        tau_cnt+=1
        times=[]
        t_new = tau
        w_list = [] 
        iter=0

        while iter<=15 and t_new not in times:
            print("\r tau {} iteration {} ".format(tau_cnt, iter), end="")
            comp_sys.protocol.t_f = t_new
    
            sim = generate_sim(comp_sys, init_state, Dev, delta_t, set_general_procs, all_state_skip=int(1/(20*delta_t)))
            sim.output = sim.run()
    
            final_state=sim.output.final_state
            sim.output.fidelity = fidelity(sim.output.trajectories)
    
            final_W = store_sys.get_potential(final_state, 0) - comp_sys.get_potential(final_state, 0)
            init_W = comp_sys.get_potential(init_state, 0) - store_sys.get_potential(init_state, 0)
            net_W = final_W + init_W
            
            sim.output.final_W = net_W
            sim.output.dt = sim.dt
            sim.output.nsteps = sim.nsteps
            sim.output.init_state = init_state
            sim.output.device = copy.deepcopy(Dev).__dict__
            sim.output.store_params = store_sys.protocol.params[:,0]
            sim.output.comp_params = comp_sys.protocol.params[:,0]
            sim.output.kT_prime =kT_prime
            write_dict['sims'].append(sim.output.__dict__)
    
            w_list.append(np.mean(net_W))
            times.append(t_new)



            if iter==0:
                curr_w = w_list[0]
                curr_t = times[0]
                t_new += tau_resolution*np.sign(t_crit-t_new)
                iter += 1
            else:
                print('t_old, t_new, w_old, w_new',curr_t, times[-1], curr_w, w_list[-1])
                t_new, [curr_t, curr_w], iter = newtime(curr_t, curr_w, times, w_list, iter, tau_resolution)
        
        write_dict['tau_list'].extend(times)

def get_best_work(sim_list, fidelity_thresh=.99):
    mean_works = [ np.mean(sim['final_W']) for sim in sim_list ]
    fids = [ sim['fidelity']['overall'] for sim in sim_list ]
    valid_works = np.array(mean_works)[np.array(fids) >= fidelity_thresh]
    min_work, index = None, None
    if len(valid_works) > 0 :
        min_work = np.min(valid_works)
        index = np.squeeze(np.where(mean_works==min_work))

    return min_work, index


        

