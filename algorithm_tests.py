import bit_flip_sweep as bfs
from kyle_tools import separate_by_state
from kyle_tools.info_space import is_bundle
import numpy as np
import matplotlib.pyplot as plt


def test_tau_candidates(N_full, N_mean, N_trial, Dev, dt=1/100, plot=False, bundle_size=5):
    i=0
    print('\r {:.0f}% done'.format(100*i/N_trial), end='')
    lcand=[[], [], []]
    rcand=[[], [], []]
    tc=[[], [], []]

    lcand_2=[[], [], []]
    rcand_2=[[], [], []]
    tc_2=[[], [], []]

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
        #print(np.shape(init_s_3)[0])


        init_s_1 = init_s_1[:N_full]
        sim_1 = bfs.generate_sim(comp, init_s_1, Dev, dt, bfs.set_mean_procs)
        sim_2 = bfs.generate_sim(comp, init_s_2, Dev, dt, bfs.set_mean_evolution_procs)
        sim_3 = bfs.generate_sim(comp, init_s_3, Dev, dt, bfs.set_bundle_evo_procs, weights=weights_3)


        print('\r trial {} of {} initialized'.format(i,N_trial), end='')
        colors=['r','b','k', 'g']
        colors_2=['m','y','c']
        for j,sim in enumerate([sim_1, sim_2, sim_3]):
            sim.output = sim.run()
            z_states = sim.output.zero_means['values']
            o_states = sim.output.one_means['values']
            times = np.linspace(0, comp.protocol.t_f, sim.nsteps+1)
            tau_candidates, t_crit = bfs.get_tau_candidate(z_states, o_states, times)
            tau_candidates_2, t_crit_2 = get_tau_candidate_new(sim)
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
                        ax[j,i].axvline(item, c=colors[k], alpha=.8)
                    for k,item in enumerate([tau_candidates_2[0], tau_candidates_2[1], t_crit_2]):
                        ax[j,i].axvline(item, c=colors_2[k], alpha=.8)
                
                if j > 0:
                    for item in [z_diff, o_diff]:
                        ax[3,i].plot(times, item[:,0,0].transpose(), c=colors[j])
                        ax[3,i].plot(times, item[:,0,1].transpose(), c=colors[j])
                        ax[3,i].plot(times, item[:,1,1].transpose(), c=colors[j])


            lcand[j].append(tau_candidates[0])
            rcand[j].append(tau_candidates[1])
            tc[j].append(t_crit)

            lcand_2[j].append(tau_candidates_2[0])
            rcand_2[j].append(tau_candidates_2[1])
            tc_2[j].append(t_crit_2)


        i+=1
        print('\r {:.0f}% done'.format(100*i/N_trial), end='')
    if plot:
        ax[0,0].set_title('full_sim')
        ax[1,0].set_title('mean_evo')
        ax[2,0].set_title('bundle_evo')
        ax[3,0].set_title('bundle:black, full:blue')
        plt.show()
    return [lcand, rcand, tc], [lcand_2, rcand_2, tc_2]


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


def get_tau_candidate_new(sim):
    burn = int(1/sim.dt)

    z_means = sim.output.zero_means['values'][burn:]
    o_means = sim.output.one_means['values'][burn:]
    t = np.linspace(0, sim.nsteps*sim.dt, sim.nsteps+1)[burn:]
    t_list =[[],[]]
    
    z_kinetic = np.sum(z_means[...,1]**2 * (.25, 1), axis=-1)
    o_kinetic = np.sum(o_means[...,1]**2 * (.25, 1), axis=-1)
    i_z = np.max(np.where(z_kinetic==z_kinetic.min()))
    i_o = np.max(np.where(o_kinetic==o_kinetic.min()))

    i_crit = int((i_z+i_o)/2)
    t_crit = (t[i_z] + t[i_o])/2


    assert np.sign(z_means[i_crit,0,0]) == 1 and np.sign(o_means[i_crit,0,0]) == -1, 'not a good swap'

    for item in [z_means, o_means]:
        t_list[1].append(find_zero(item[i_crit:,1,1], t[i_crit:])[1])
        dt_left = find_zero(item[i_crit::-1,1,1],-t[i_crit::-1]+t[i_crit], mode='increasing')[1]
        t_list[0].append(t[i_crit]-dt_left)

    for i,item in enumerate(t_list):
        t_list[i] = np.mean(item)
    
    return t_list, t_crit
        
#from FQ_sympy_functions import DeviceParams
#Devc = DeviceParams()
#L,R,T = test_tau_candidates(5_000, 5_000, 4, Devc, dt=1/100, plot=True, bundle_size=4)