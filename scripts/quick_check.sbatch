#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 3
#SBATCH --time 3:00:00
#SBATCH --partition=hns

ml julia/1.3.1
julia ./quick_check.jl
