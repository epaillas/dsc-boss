#!/bin/bash
#SBATCH --account=desi
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -c 256
#SBATCH --array=0-20

REGION=NGC
START_PHASE=1
N_PHASE=100
# START_PHASE=$((SLURM_ARRAY_TASK_ID * N_PHASE + 1))
WEIGHT_TYPE=default_FKP
ZMIN=0.45
ZMAX=0.6
ZPAD=0.02
NTHREADS=256

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=$NTHREADS
export NUMEXPR_MAX_THREADS=$NTHREADS
export OMP_PLACES=threads
CODE_DIR=/global/u1/e/epaillas/code/dsc-boss

srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi -c 256 \
python $CODE_DIR/ds_patchy.py \
--region $REGION \
--start_phase $START_PHASE \
--n_phase $N_PHASE \
--weight_type $WEIGHT_TYPE \
--save_clustering \
--save_quantiles \
--save_density \
--add_redshift_padding \
--zpad $ZPAD \
--zmin $ZMIN \
--zmax $ZMAX \
--nthreads $NTHREADS \