import numpy as np
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.expanduser('~/source'))

from kyle_tools import w_TUR, open_json
from kyle_tools.visualization import pcolor_diagram, heatmap, annotate_heatmap
from matplotlib.colors import LogNorm



def plot_tau_sweep_sim_results(output_dict):
    sim_results = output_dict['sim_results']
    kT_prime = output_dict['kT_prime']
    _, ax = plt.subplots(len(sim_results),2) 
    for i, item in enumerate(sim_results):
        plot_sim_avg(item['prelim_sim'], ax[i,0], highlight_times=item['tau_list'])
        works = []
        std = []
        for sim in item['sims']:
            final_W = np.multiply(sim['final_W'], 1/kT_prime)
            works.append(np.mean(final_W))
            std.append(np.std(final_W))
        min_W = np.min(works)
        ax[i,1].errorbar(item['tau_list'], works, yerr=std, fmt="o")
        ax[i,1].set_title('L={:.2e},min_W={:.2f}'.format(sim['device']['L'], min_W))
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
    ax.axvline(sim_dict['t_crit'], c='k') 
    for t_val in sim_dict['tau_list']:
        ax.axvline(t_val, c='r')
    if highlight_times is not None:
        for t_val in highlight_times:
            ax.axvline(t_val, alpha=.4)
    ax.set_xlabel('t/sqrt(LW)')

def work_heatmap(directory, key1, key2):
    '''
    returns sweep values for keys 1 and 2, and work values in units of kT_prime
    '''
    best_sims = []
    directory_list = os.listdir(directory)
    for param_sweep in directory_list:
        current_sweep = open_json(directory+param_sweep)
        kT_prime = current_sweep['kT_prime']
        for param_value in current_sweep['sim_results']:
            temp_dict={}

            try:
                temp_dict['work'] = param_value['min_work']
            except:
                continue
            temp_dict[key1] = None
            temp_dict[key2] = None
            if temp_dict['work'] is not None:
                best_sim = param_value['sims'][param_value['min_work_index']]
                temp_dict[key1] = best_sim['device'][key1]
                temp_dict[key2] = best_sim['device'][key2]
                temp_dict['work'] = temp_dict['work']/kT_prime
            best_sims.append(temp_dict)

    axis1 = list(set([ sim[key1] for sim in best_sims if sim[key1] is not None]))
    axis2 = list(set([ sim[key2] for sim in best_sims if sim[key2] is not None]))
    for lst in [axis1,axis2]:
        lst.sort()

    W = np.zeros((len(axis1), len(axis2)))

    for ix, xval in enumerate(axis1):
        for iy, yval in enumerate(axis2):
            valid_works= [ item['work'] for item in best_sims if item[key1]==xval and item[key2]==yval ]
            valid_works = [item for item in valid_works if item is not None]
            W[ix, iy] = np.nan
            if len(valid_works)>0:
                W[ix,iy] = np.min(valid_works)

    return(axis1, axis2, W)

def plot_work_heatmap(dir, key1, key2, ax=None, **imshow_kwargs):
    x, y, W = work_heatmap(dir, key1, key2)
    max_w = np.nanmax(W)
    min_w = np.nanmin(W)

    
    x = ['{:.2e}'.format(val) for val in x]
    y = ['{:.2e}'.format(val) for val in y]

    if ax is None:
        fig, ax = plt.subplots()

    im, cbar = heatmap(W, x, y, ax=ax, cmap="plasma", cbarlabel="work (k_B T)", **imshow_kwargs, norm=LogNorm(vmin=min_w, vmax=max_w))

    ax.set_xlabel(key1)
    ax.set_ylabel(key2)


    texts = annotate_heatmap(im, valfmt="{x:.2f}")

    fig.tight_layout()

    return ax, cbar




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
        

def plot_all_state(output_dict, which_sims=np.s_[:]):
    kT_prime = output_dict['kT_prime']
    sim_results = output_dict['sim_results']
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
                w = np.mean(sim['final_W'])/kT_prime
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













