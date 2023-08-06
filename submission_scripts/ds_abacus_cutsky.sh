#!/bin/bash

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=256
export OMP_PLACES=threads
export NUMEXPR_MAX_THREADS=256
CODE_DIR=/global/u1/e/epaillas/code/dsc-boss

START_PHASE=0
N_PHASE=25
START_HOD=26
N_HOD=1

srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi -c 256 \
    python $CODE_DIR/ds_abacus_cutsky.py \
    --start_phase $START_PHASE \
    --n_phase $N_PHASE \
    --start_hod $START_HOD \
    --n_hod $N_HOD \
    --save_clustering \
    --save_quantiles \
    --save_density \
    --add_redshift_padding \