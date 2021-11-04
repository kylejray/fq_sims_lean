import bit_flip_sweep as bfs
import numpy as np
import matplotlib.pyplot as plt

def test_tau_candidates(N_full, N_mean, N_trial, Dev, plot=False):
    i=0
    lcand=[[],[]]
    rcand=[[],[]]
    tc=[[],[]]
    while i<N_trial:
        print('\r {:.0f}% done'.format(100*i/N_trial), end='')
        store, comp = bfs.set_systems(Dev, eq_tau=1, comp_tau=10, d_comp_store=.2)
        init_s_1 = bfs.generate_eq_state(store, N_mean, bfs.kT_prime)
        init_s_2 = bfs.info_state_means(init_s_1)
        init_s_1 = init_s_1[:N_full]
        sim_1 = bfs.generate_sim(comp, init_s_1, Dev, 1/100, bfs.set_mean_procs)
        sim_2 = bfs.generate_sim(comp, init_s_2, Dev, 1/100, bfs.set_mean_evolution_procs)
        if plot:
            fig, ax = plt.subplots(2)

        for j,sim in enumerate([sim_1, sim_2]):
            sim.output = sim.run()
            z_states = sim.output.zero_means['values']
            o_states = sim.output.one_means['values']
            times = np.linspace(0, comp.protocol.t_f, sim.nsteps+1)
            tau_candidates, t_crit = bfs.get_tau_candidate(z_states, o_states, times)
            if plot:
                for item in [z_states, o_states]:
                    ax[j].plot(times, item[:,0,0].transpose())
                    ax[j].plot(times, item[:,0,1].transpose())
                    ax[j].plot(times, item[:,1,1].transpose())
                    for item in [tau_candidates[0], tau_candidates[1], t_crit]:
                        ax[j].axvline(item)

            lcand[j].append(tau_candidates[0])
            rcand[j].append(tau_candidates[1])
            tc[j].append(t_crit)
        i+=1
    if plot:
        ax[0].set_title('full_sim')
        ax[1].set_title('mean_evo')
        plt.show()
    return lcand, rcand, tc