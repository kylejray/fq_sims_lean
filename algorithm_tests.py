import bit_flip_sweep as bfs
from kyle_tools import separate_by_state
import numpy as np
import matplotlib.pyplot as plt
from FQ_sympy_functions import DeviceParams

def test_tau_candidates(N_full, N_mean, N_trial, Dev, dt=1/100, plot=False, bundle_size=5):
    i=0
    print('\r {:.0f}% done'.format(100*i/N_trial), end='')
    lcand=[[], [], []]
    rcand=[[], [], []]
    tc=[[], [], []]
    if plot:
        fig, ax = plt.subplots(4,N_trial)
    while i<N_trial:
        if plot:
            Dev.perturb()

        store, comp = bfs.set_systems(Dev, eq_tau=1, comp_tau=10, d_store_comp=[.2,.2])
        init_s_1 = bfs.generate_eq_state(store, N_mean, bfs.kT_prime)
        init_s_2 = bfs.info_state_means(init_s_1)
        is_bools = separate_by_state(init_s_1[...,0,0])

        init_s_3, weights_3 = is_bundle(init_s_1, is_bools, bundle_size)
        print(np.shape(init_s_3)[0])


        init_s_1 = init_s_1[:N_full]
        sim_1 = bfs.generate_sim(comp, init_s_1, Dev, dt, bfs.set_mean_procs)
        sim_2 = bfs.generate_sim(comp, init_s_2, Dev, dt, bfs.set_mean_evolution_procs)
        sim_3 = bfs.generate_sim(comp, init_s_3, Dev, dt, bfs.set_bundle_evo_procs, weights=weights_3)


        print('\r trial {} of {} initialized'.format(i,N_trial), end='')
        colors=['r','b','k', 'g']
        for j,sim in enumerate([sim_1, sim_2, sim_3]):
            sim.output = sim.run()
            z_states = sim.output.zero_means['values']
            o_states = sim.output.one_means['values']
            times = np.linspace(0, comp.protocol.t_f, sim.nsteps+1)
            tau_candidates, t_crit = bfs.get_tau_candidate(z_states, o_states, times)
            if plot:
                if j == 0:
                    full_z = z_states
                    full_o = o_states
                z_diff = z_states - full_z
                o_diff = o_states - full_o

                for item in [z_states, o_states]:
                    ax[j,i].plot(times, item[:,0,0].transpose())
                    ax[j,i].plot(times, item[:,0,1].transpose())
                    ax[j,i].plot(times, item[:,1,1].transpose())
                    for k,item in enumerate([tau_candidates[0], tau_candidates[1], t_crit]):
                        ax[j,i].axvline(item, c=colors[k])
                
                if j > 0:
                    for item in [z_diff, o_diff]:
                        ax[3,i].plot(times, item[:,0,0].transpose(), c=colors[j])
                        ax[3,i].plot(times, item[:,0,1].transpose(), c=colors[j])
                        ax[3,i].plot(times, item[:,1,1].transpose(), c=colors[j])


            lcand[j].append(tau_candidates[0])
            rcand[j].append(tau_candidates[1])
            tc[j].append(t_crit)
        i+=1
        print('\r {:.0f}% done'.format(100*i/N_trial), end='')
    if plot:
        ax[0,0].set_title('full_sim')
        ax[1,0].set_title('mean_evo')
        ax[2,0].set_title('bundle_evo')
        ax[3,0].set_title('bundle:black, full:blue')
        plt.show()
    return lcand, rcand, tc

def eq_dist_bundle(state_ensemble, is_bools, eq_sys, N):
    output = None
    for key in is_bools:
        info_state = state_ensemble[is_bools[key]][:N]
        prob = eq_sys.get_energy(info_state, 0)
        info_state_weight = np.exp(-prob)
        if output is None:
            output = info_state
            weights= info_state_weight
        else:
            output = np.append(output, info_state, axis=0)
            weights = np.append(weights, info_state_weight, axis=0)
        

    return output[weights!=0], weights[weights!=0]

def is_bundle(state_ensemble, is_bools, n_points):
    coords=[]
    weights=[]
    for key in is_bools:
        info_state = state_ensemble[is_bools[key]]
        c, w = representative_bundle(info_state, n_points)
        coords.append(c)
        weights.append(w)
    output = coords[0]
    out_weight = weights[0]
    for item in zip(coords[1:], weights[1:]):
        output = np.append(output,item[0], axis=0)
        out_weight = np.append(out_weight, item[1], axis=0)
    
    return output, out_weight

def representative_bundle(state_ensemble, n_points):
    state_ensemble =np.squeeze(state_ensemble)
    shape = np.shape(state_ensemble)
    N, state_shape = shape[0], shape[1:]
    state = state_ensemble.reshape(N, np.prod(state_shape))


    state_means = np.mean(state, axis=0)
    state_stds = np.std(state, axis=0)
    bins = np.linspace(state_means-3*state_stds, state_means+3*state_stds, n_points+1).transpose()
    hist = np.histogramdd(state, bins=bins, density=True)
    values = bins[:,:-1] + np.diff(bins)/2
    mesh = np.meshgrid(*values)
    mesh = [item.ravel() for item in mesh]
    coords = np.array([ np.reshape(item, state_shape) for item in list(zip(*mesh))])
    weights = np.ravel(hist[0])

    return coords[weights != 0], weights[weights != 0]

#Devc = DeviceParams()
#L,R,T = test_tau_candidates(5_000, 5_000, 4, Devc, dt=1/100, plot=True, bundle_size=4)