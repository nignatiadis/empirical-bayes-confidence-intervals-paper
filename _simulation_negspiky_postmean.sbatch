#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=34:00:00
#SBATCH --ntasks-per-node=14
#SBATCH --output=negspiky_postmean.log
#SBATCH --partition=stat,hns,normal

# load modules 
ml load julia/1.6.2

# execute script
julia -t 15 simulation_ebci.jl postmean negspiky
