#!/bin/sh
#SBATCH --mem-per-cpu=15G

python ../scripts/python/train_svm.py $SLURM_ARRAY_TASK_ID