# add test_things.py module to path
import sys
import run_sweep as rs
import numpy as np
import math

# always include these lines of MPI code!
from mpi4py import MPI
comm = MPI.COMM_WORLD
size = comm.Get_size()  # number of MPI procs
rank = comm.Get_rank()  # i.d. for local proc


N_test=40_000
dsci = [.16,.31]

run_name = 'heatmap_sym_detail_2X20L_5X8g_p16p31_N40/'
save_dir = '/home/kylejray/FQ_sims/results/{}/'.format(run_name)

rs.Dev.change_vals({'dbeta': .02})

g_range = np.linspace(8,25, 80)
max_g = 5
g_lists = [ g_range[i*max_g:(i+1)*max_g] for i in range(math.ceil(len(g_range)/max_g))]

L_range = np.linspace(4E-10, 5E-10, 40)
# max_L sets the max number of jobs that will be requested
max_L = 20
L_lists = [ L_range[i*max_L:(i+1)*max_L] for i in range(math.ceil(len(L_range)/max_L))]


for L_values in L_lists:
   # if you have an external file with parameters as function of rank:

   # params = load(param_file) # e.g. np.load(params.npy) or load from csv, etc.
   # my_param = params[rank]

   my_L = L_values[rank]
   rs.Dev.change_vals({'L': my_L})
   # save parameters to output file by printing
   sys.stdout.write('my rank:{} of {}, dev_L={} '.format(rank+1, size, rs.Dev.L))

   # perform local computation using local param
   for g_list in g_lists:
      g_dict = {'param': 'gamma', 'sweep': g_list}
      d = rs.run_sweep(sweep_dict=g_dict, N_t=N_test, save_dir=save_dir, d_s_c_init=dsci)
      sys.stdout.flush()

   # save your results

   '''
   # add a comm barrier so that param printing finishes first
   # comm.barrier()

   # a simple example of cross-node computation using mpi
   # global_array = np.zeros(5)
   # comm.Allreduce(my_array, global_array, op=MPI.SUM)

   # record the result in the output file
   # if rank == 0:
      #  print('global array sum: ', global_array)
   '''
