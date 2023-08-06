#!/bin/bash
#SBATCH --account=desi_g
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --gpus 1
#SBATCH --constraint=gpu
#SBATCH --array=0-0


REGION=SGC
N_PHASE=100
START_PHASE=1
# START_PHASE=$((SLURM_ARRAY_TASK_ID * N_PHASE + 1))
WEIGHT_TYPE=default_FKP
ZMIN=0.45
ZMAX=0.6
NTHREADS=4

source /global/common/software/desi/users/adematti/cosmodesi_environment.sh main
export OMP_NUM_THREADS=$NTHREADS
export NUMEXPR_MAX_THREADS=$NTHREADS
export OMP_PLACES=threads
CODE_DIR=/global/u1/e/epaillas/code/dsc-boss

srun -N 1 -C gpu -t 04:00:00 --gpus 1 --qos interactive --account desi_g \
python $CODE_DIR/tpcf_patchy.py \
--region $REGION \
--start_phase $START_PHASE \
--n_phase $N_PHASE \
--weight_type $WEIGHT_TYPE \
--zmin $ZMIN \
--zmax $ZMAX \
--nthreads $NTHREADS \
--use_gpu \
