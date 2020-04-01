#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 2
#SBATCH --time 30:0)
#SBATCH --partition=hns

ml julia/1.3.1
julia ./quick_check.jl