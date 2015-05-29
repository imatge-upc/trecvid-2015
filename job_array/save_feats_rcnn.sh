#!/bin/sh
#SBATCH --mem-per-cpu=6G

python ../scripts/python/save_feats.py $SLURM_ARRAY_TASK_ID