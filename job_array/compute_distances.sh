#!/bin/sh
#SBATCH --mem-per-cpu=6G

python ../scripts/python/compute_distances.py $SLURM_ARRAY_TASK_ID