#!/bin/sh
#SBATCH --mem-per-cpu=10G

python ../scripts/python/merge.py $SLURM_ARRAY_TASK_ID
