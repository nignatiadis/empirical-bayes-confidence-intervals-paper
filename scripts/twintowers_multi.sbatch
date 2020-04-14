#!/bin/bash
#SBATCH --job-name=array
#SBATCH --array=1-10
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --partition=hns

ml julia/1.3.1
julia ./twintowers_multi_run.jl $SLURM_ARRAY_TASK_ID
