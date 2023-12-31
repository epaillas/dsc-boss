#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH --constraint=cpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH --array=0-4,13

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=256
export OMP_PLACES=threads
export NUMEXPR_MAX_THREADS=256

N_HOD=100
START_HOD=0
N_COSMO=1
START_COSMO=$((SLURM_ARRAY_TASK_ID * N_COSMO))

time python /global/homes/e/epaillas/code/dsc-boss/ds_abacus_cubic.py --start_hod "$START_HOD" --n_hod "$N_HOD" --start_cosmo "$START_COSMO" --n_cosmo "$N_COSMO"
