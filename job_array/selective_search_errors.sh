#!/bin/sh
#SBATCH --mem-per-cpu=6G

matlab -nodisplay '+single_thread+' '+jvm_string+' -nodesktop -r "addpath('../scripts/matlab/');, selective_search_errors(${SLURM_ARRAY_TASK_ID});"

