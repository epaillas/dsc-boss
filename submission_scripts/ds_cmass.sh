#!/bin/bash

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=256
export OMP_PLACES=threads
export NUMEXPR_MAX_THREADS=256
CODE_DIR=/global/u1/e/epaillas/code/dsc-boss

WEIGHT_TYPE=default_FKP

srun -N 1 -C cpu -t 04:00:00 --qos interactive --account desi -c 256 \
    python $CODE_DIR/ds_cmass.py \
    --weight_type $WEIGHT_TYPE \
    --save_clustering \
    --save_quantiles \
    --save_density \
    --add_redshift_padding \