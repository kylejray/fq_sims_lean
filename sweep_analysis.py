import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.expanduser('~/source'))

from kyle_tools import w_TUR, open_json

from kyle_tools.visualization import pcolor_diagram, heatmap, annotate_heatmap
from kyle_tools.utilities import file_list
from matplotlib.colors import LogNorm



def plot_tau_sweep_sim_results(output_dict):
    sim_results = output_dict['sim_results']
    _, ax = plt.subplots(len(sim_results),2) 
    for i, item in enumerate(sim_results):
        plot_sim_avg(item['prelim_sim'], ax[i,0], highlight_times=item['tau_list'])
        works = []
        std = []

        for sim in item['sims']:
            final_W = sim['final_W']
            works.append(np.mean(final_W))
            std.append(np.std(final_W))
        min_W = item['min_work']

        ax[i,1].errorbar(item['tau_list'], works, yerr=std, fmt="o")
        ax[i,1].set_title('L={:.2e},min_W={:.2f}'.format(sim['device']['L'], min_W))
        ax[i,1].axhline(min_W)

def plot_tau_sweep(output_dict, param_name='L'):
    sim_results = output_dict['sim_results']
    _, ax = plt.subplots(len(sim_results),2) 
    for i, item in enumerate(sim_results):
        tau_sweep=item['tau_sweep']

        plot_sim_avg(item['prelim_sim'], ax[i,0], highlight_times=item['tau_list'])

        N_trials=len(tau_sweep['sim']['initial_state'])

        works= tau_sweep['mean_W']
        std = [ 3*std/np.sqrt(N_trials) for std in tau_sweep['std_W']]
        min_W = item['min_work']

        ax[i,1].errorbar(item['tau_list'], works, yerr=std, fmt="o")
        ax[i,1].set_title(param_name+'={:.2e},min_W={:.2f}'.format(tau_sweep['sim']['device'][param_name], min_W))
        ax[i,1].axhline(min_W)

def plot_sim_avg(sim_dict, ax, highlight_times=None):
    zm = np.array(sim_dict['zero_means']['values'])[...,1]
    om = np.array(sim_dict['one_means']['values'])[...,1]
    zmp = np.array(sim_dict['zero_means']['values'])[...,0,0]
    omp = np.array(sim_dict['one_means']['values'])[...,0,0]
    time = np.linspace(0, sim_dict['dt']*sim_dict['nsteps'],sim_dict['nsteps']+1)
    for item in zm.transpose():
        ax.plot(time, item)
    for item in om.transpose():
        ax.plot(time, item)
    ax.plot(time, zmp)
    ax.plot(time, omp)
    if 't_crit' in sim_dict.keys():
        ax.axvline(sim_dict['t_crit'], c='k') 
    if 'tau_list' in sim_dict.keys():
        for t_val in sim_dict['tau_list']:
            ax.axvline(t_val, c='r')
    if highlight_times is not None:
        for t_val in highlight_times:
            ax.axvline(t_val, alpha=.1)
    ax.set_xlabel('t/sqrt(LW)')

def get_tau_sweep_if_states(sim_results):
    init_states = []
    final_states = []
    for sweep in sim_results:
        if 'terminated' in sweep.keys():
            init_states.append(sweep['prelim_sim']['initial_state'])
            final_states.append(None)
        else:
            sim = sweep['tau_sweep']['sim']
            if_states = [ np.array(np.array(sim[item]))[...,0,0] for item in ['initial_state', 'final_state'] ]
            init_states.append(if_states[0])
            final_states.append(if_states[1])
    return init_states, final_states



def plot_sim_avg_err(times, state_list, slice_list, err_scale, ax, kwarg_list=None, **plt_kwargs):
    i=0
    for mean_dict in state_list:

        vals = np.array(mean_dict['values'])
        try:
            N = sum(mean_dict['trial_indices'])
        except:
            N = 1
        
        try:
            errs = err_scale * np.sqrt(N) * np.array(mean_dict['std_error'])
        except:
            print('had a problem determining errors')
            errs = np.zeros(vals.shape)

        for item in slice_list:
            v, err = vals[item], errs[item]
            if kwarg_list is not None:
                plt_kwargs.update(kwarg_list[i])

            ax.errorbar(times, v, yerr=err, **plt_kwargs)

            if i < len(slice_list)*len(state_list):
                i += 1

def highlight_times(t_list, opacity, ax):
    for t_val in t_list:
        ax.axvline(t_val, alpha=opacity)

def plot_one_allstate(all_state, ax, state_slice=np.s_[...,0,0], t=None, sim_dict=None):
    # if t is None, sim_dict cannot be None
    step_indices = eval(all_state['step_indices'])
    states = np.array(all_state['states'])[state_slice]

    if t is None:
        if sim_dict is not None:
            t = np.linspace(0, sim_dict['dt'] * sim_dict['nsteps'], sim_dict['nsteps']+1)[step_indices]
        if sim_dict is None:
            t= np.linspace(0, 1, len(step_indices)+1)
    else:
        t = np.array(t)[step_indices]
    
    ax.plot(t, states.transpose())


        
        



def work_heatmap(directory, key1='L', key2='gamma', fidelity_thresh=.99):
    '''
    returns sweep values for keys 1 and 2, and work values in units of kT_prime
    '''
    best_sims = []
    directory_list = file_list(directory, extension_list=['.json'])
    for param_sweep in directory_list:
        current_sweep = open_json(directory+param_sweep)
        for param_value in current_sweep['sim_results']:
            temp_dict={}

            try:
                temp_dict['work'] = param_value['min_work']
                best_idx = param_value['min_work_index']
            except:
                continue
            temp_dict[key1] = None
            temp_dict[key2] = None

            try:
                tau_sweep = param_value['tau_sweep']
            except:
                continue

            temp_dict[key1], temp_dict[key2] = [tau_sweep['sim']['device'][key] for key in [key1,key2]]
            temp_dict['fidelity'] = tau_sweep['fidelity']['overall'][best_idx]
            temp_dict['valid_final_state']= tau_sweep['valid_final_state'][best_idx]
            
            idx = np.argmin(tau_sweep['mean_W'])
            abs_min = {}
            abs_min['min_W'] = tau_sweep['mean_W'][idx]
            abs_min['valid_fs'] = tau_sweep['valid_final_state'][idx]
            abs_min['fidelity'] = tau_sweep['fidelity']['overall'][idx]
            temp_dict['amw'] = abs_min

            best_sims.append(temp_dict)

    axis1 = list(set([ sim[key1] for sim in best_sims if sim[key1] is not None]))
    axis2 = list(set([ sim[key2] for sim in best_sims if sim[key2] is not None]))
    for lst in [axis1,axis2]:
        lst.sort()

    W= np.zeros((len(axis1), len(axis2)))
    W[:] = np.nan
    AMW_dict = { key:np.empty((len(axis1), len(axis2))) for key in ['min_W', 'valid_fs', 'fidelity'] }
    for key, val in AMW_dict.items():
        val[:] = np.nan


    for ix, xval in enumerate(axis1):
        for iy, yval in enumerate(axis2):

            #F[ix,iy] = min([ item['fidelity'] for item in best_sims if item[key1]==xval and item[key2]==yval ])
            #VFS[ix,iy] = min([ item['valid_final_state'] for item in best_sims if item[key1]==xval and item[key2]#==yval ])

            valid_works= [ item['work'] for item in best_sims if item[key1]==xval and item[key2]==yval ]
            valid_works = [item for item in valid_works if item is not None]
            if len(valid_works)>0:
                W[ix,iy] = np.min(valid_works)

            abs_works= [ item['amw'] for item in best_sims if item[key1]==xval and item[key2]==yval ]
            if len(abs_works) == 0:
                continue
            if len(abs_works) > 1:
                minW = np.min([item['min_W'] for item in abs_works])
                abs_works = list(filter(lambda x: x['min_W']==minW, abs_works))
            for key, value in AMW_dict.items():
                value[ix, iy] = abs_works[0][key]

    return axis1, axis2, W, AMW_dict

def plot_work_heatmap(heatmap_vals, key1, key2, ax=None, cbar_range=None, label=True, label_data=None, label_fmt="{x}", label_color=None, **imshow_kwargs):
    x, y, W, _ = heatmap_vals

    if cbar_range is None:
        max_w = np.nanmax(W)
        min_w = np.nanmin(W)
    else:
        min_w, max_w = cbar_range

    
    x = ['{:.1e}'.format(val) for val in x]
    y = ['{:.1f}'.format(val) for val in y]

    if ax is None:
        fig, ax = plt.subplots()


    im, cbar = heatmap(W, x, y, ax=ax, cmap="plasma", cbarlabel="work ($k_B$ T)", **imshow_kwargs, norm=LogNorm(vmin=min_w, vmax=max_w), cbar_kw={'shrink':.5})

    ax.set_xlabel(key1)
    ax.set_ylabel(key2)

    if label:
        annotate_kw={}
        annotate_kw.update(textcolors='white')
        if label_color is not None:
            annotate_kw.update(textcolors=label_color)

        annotate_kw.update(valfmt=label_fmt)    
        texts = annotate_heatmap(im, data=label_data, **annotate_kw)
    try:
        fig.tight_layout()
    except:
        pass

    return ax, cbar, [x,y,W]


def heatmap_label(W, label_names, rules):
    labels = np.empty(W.shape, dtype='object')
    labels[...] = ' '
    for label, rule in zip(label_names, rules):
        e_rule = eval(rule.format('W'))
        labels[e_rule] = label
    return labels


def get_best_protocols(output_dict, keys=[]):
    valid_sims = [('terminated' not in sim.keys()) and (sim['min_work'] is not None) for sim in output_dict['sim_results']]

    param_sweep = [item for i,item in enumerate(output_dict['param_sweep']) if valid_sims[i]]

    sims = [item for i, item in enumerate(output_dict['sim_results']) if valid_sims[i]]

    idx = [sim['min_work_index'] for sim in sims]

    output={}

    keys = keys + ['fidelity', 'device', 'final_W']
    for key in keys:
        output[key] = [ sim['sims'][idx[i]][key] for i,sim in enumerate(sims)]

    output['param'] = param_sweep

    return output


def plot_w_TUR(out_d, output=None, ax=None, yscale='log'):
    ft_bound = []
    min_bound = []
    guarnieri_bound = []
    scaled_var = []
    work = []
    for item in out_d['sim_results']:
        if 'terminated' not in item.keys():
            for sim in item['sims']:
                work_list = sim['final_W']
                W = np.mean(work_list)

                bnd, var = w_TUR(sim['init_state'], sim['final_state'], W)
                min_bound.append(1/np.mean(np.tanh(np.divide(work_list,2)))-1)
                work.append(W)
                guarnieri_bound.append((1/np.sinh(inv_xtanhx(W/2))**2))
                ft_bound.append(bnd)
                scaled_var.append(var)
    if ax is None:
        _, ax = plt.subplots()
    scaled_var = np.array(scaled_var)  
    ax.scatter(work, ft_bound, c='r')
    ax.scatter(work, scaled_var[:,0], c='b')
    ax.scatter(work, scaled_var[:,1], c='g')
    ax.scatter(work, np.divide(2,work), c='r', marker='+')
    ax.scatter(work, min_bound, c='k', marker='^', alpha=.5 )
    ax.scatter(work, guarnieri_bound, c='r', marker='v')
    ax.legend(['FT_bound','phi_var', 'phi_dc_var', 'linear_bound', 'min_bound', 'guarnieri_bound'])
    ax.set_xlabel('avg_work')
    ax.set_yscale(yscale)

    if output is not None:
        return work, bound, scaled_var
        

def plot_all_state(output_dict, which_sims=np.s_[:], which_params=np.s_[:]):
    sim_results = np.asarray(output_dict['sim_results'])[which_params]
    n_columns = len(sim_results)
    ax_j =0 
    for param_set in sim_results:
        decider= np.array(range(0,len(param_set['sims'])))[which_sims]
        if ax_j == 0:
            _, ax = plt.subplots(len(decider),n_columns)
        ax_i=0
        for i, sim in enumerate(param_set['sims']):
            if i in decider:
                all_state = sim['all_state']
                all_s = np.array(all_state['states'])
                step_indices = eval(all_state['step_indices'])
                t = np.linspace(0, sim['dt'] * sim['nsteps'], sim['nsteps']+1)[step_indices]
                w = np.mean(sim['final_W'])
                ax[ax_i,ax_j].plot(t, all_s[...,0,0].transpose())
                ax[ax_i,ax_j].set_title('W={:.3f},F={:.3f}'.format(w,sim['fidelity']['overall']))

                ax_i+=1
        ax_j += 1

def inv_xtanhx(arg, tol=.001):
    done=False
    steps = int(1/tol)
    i=0
    while not done and i <= 5:
        x = np.linspace(arg-3, arg+3, 7*steps)
        x = x[x>0]
        y = x * np.tanh(x) - arg
        absy = np.sign(y)
        decider = np.diff(absy, prepend=absy[0])
        i_g = np.where(decider>0)[0]
        assert len(i_g)==1, 'inv xtanx  algorithm failed'
        i_l = i_g-1
        ratio = abs(y[i_g]/y[i_l])
        output = (ratio*x[i_l] +x[i_g])/(ratio+1)

        if np.isclose(output, arg*np.tanh(arg), atol=tol):
            done=True
        else:
            steps = steps*2
            i+=1
            
    return output 













