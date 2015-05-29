#!/bin/sh
#SBATCH --mem-per-cpu=6G

matlab -nodisplay '+single_thread+' '+jvm_string+' -nodesktop -r "addpath('../scripts/matlab/');, run_selective_search(${SLURM_ARRAY_TASK_ID});"

