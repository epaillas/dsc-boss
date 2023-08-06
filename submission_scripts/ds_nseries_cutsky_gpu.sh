#!/bin/bash
#SBATCH --account=desi_g
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus 1
#SBATCH --constraint=gpu

START_PHASE=1
N_PHASE=85
WEIGHT_TYPE=default
ZMIN=0.47
ZMAX=0.55
ZPAD=0.01
NTHREADS=4

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=$NTHREADS
export NUMEXPR_MAX_THREADS=$NTHREADS
export OMP_PLACES=threads
CODE_DIR=/global/u1/e/epaillas/code/dsc-boss

srun -N 1 -C gpu -t 04:00:00 --gpus 1 --qos interactive --account desi_g \
    python $CODE_DIR/ds_nseries_cutsky_exp.py \
    --start_phase $START_PHASE \
    --n_phase $N_PHASE \
    --save_clustering \
    --save_quantiles \
    --save_density \
    --zpad $ZPAD \
    --zmin $ZMIN \
    --zmax $ZMAX \
    --nthreads $NTHREADS \
    --use_gpu \
    --add_redshift_padding \
    --flatten_nz \
    # --weight_type $WEIGHT_TYPE \