#!/bin/bash

#SBATCH --job-name=full_heatmap
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH -p parallel
#SBATCH -o /home/%u/FQ_sims/results/%j.out
#SBATCH -e /home/%u/FQ_sims/results/%j.err
##SBATCH --mail-type=END,FAIL
##SBATCH --mail-user=kylejray@gmail.com

source activate ../envs

mpirun -bootstrap slurm -n 20 python run_cluster_sim.py
#srun -n 4 python run_cluster_sim.py