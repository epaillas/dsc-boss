#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH --constraint=cpu
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --array=0-4,13,100-126,130-181

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=256
export OMP_PLACES=threads
export NUMEXPR_MAX_THREADS=256

N_HOD=100
START_HOD=0
N_COSMO=1
START_COSMO=$((SLURM_ARRAY_TASK_ID * N_COSMO))
START_COSMO=0

time python ../ds_abacus_cubic.py \
    --start_hod "$START_HOD" \
    --n_hod "$N_HOD" \
    --start_cosmo "$START_COSMO" \
    --n_cosmo "$N_COSMO" \
    --nquantiles 9 \
    --outdir "/pscratch/sd/c/cuesta/ds_boss/" \
    --save_clustering \
    --save_density \
    --quantiles_clustering 0 8 \
