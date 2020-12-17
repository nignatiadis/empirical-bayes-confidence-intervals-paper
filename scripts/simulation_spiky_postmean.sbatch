#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=34:00:00
#SBATCH --ntasks-per-node=14
#SBATCH --output=spiky_postmean.log
#SBATCH --partition=stat,hns,normal

# load modules 
ml load julia/1.5.1

# execute script
julia -t 14 simulation_spiky_postmean.jl
