#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH --constraint=cpu
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH --array=0-20

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=256
export OMP_PLACES=threads
export NUMEXPR_MAX_THREADS=256

n_per_submit=100
start_idx=$((SLURM_ARRAY_TASK_ID * n_per_submit + 1))
echo $start_idx

python $HOME/code/dsc-boss/tpcf_patchy.py --start_phase "$start_idx" --n_phase "$n_per_submit"
