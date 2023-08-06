#!/bin/bash
#SBATCH --nodes=1
#SBATCH --account=desi
#SBATCH --ntasks-per-node=1
#SBATCH -c 256
#SBATCH --constraint=cpu
#SBATCH -q regular
#SBATCH -t 12:00:00

START_PHASE=1
N_PHASE=85
WEIGHT_TYPE=default
ZMIN=0.47
ZMAX=0.55
ZPAD=0.01
NTHREADS=256

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=$NTHREADS
export NUMEXPR_MAX_THREADS=$NTHREADS
export OMP_PLACES=threads
CODE_DIR=/global/u1/e/epaillas/code/dsc-boss

# srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi -c 256 \
python $CODE_DIR/ds_nseries_cutsky_exp.py \
    --start_phase $START_PHASE \
    --n_phase $N_PHASE \
    --weight_type $WEIGHT_TYPE \
    --save_clustering \
    --save_quantiles \
    --add_redshift_padding \
    --zpad $ZPAD \
    --flatten_nz \
    --zmin $ZMIN \
    --zmax $ZMAX \
    --nthreads $NTHREADS \