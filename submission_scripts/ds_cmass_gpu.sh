#!/bin/bash
#SBATCH --account=desi_g
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --gpus 1
#SBATCH --constraint=gpu

REGION=SGC
WEIGHT_TYPE=default_FKP
ZMIN=0.45
ZMAX=0.6
ZPAD=0.02
NTHREADS=4

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=$NTHREADS
export NUMEXPR_MAX_THREADS=$NTHREADS
export OMP_PLACES=threads
CODE_DIR=/global/u1/e/epaillas/code/dsc-boss

srun -N 1 -C gpu -t 04:00:00 --gpus 1 --qos interactive --account desi_g \
    python $CODE_DIR/ds_cmass.py \
    --region $REGION \
    --weight_type $WEIGHT_TYPE \
    --save_clustering \
    --save_quantiles \
    --save_density \
    --zmin $ZMIN \
    --zmax $ZMAX \
    --nthreads $NTHREADS \
    --use_gpu \
    --add_redshift_padding \
    --zpad $ZPAD \