#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 12
#SBATCH --time 28:00:00
#SBATCH --partition=hns

ml julia/1.3.1
julia ./twintowers.jl