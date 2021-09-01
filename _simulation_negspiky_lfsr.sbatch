#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=34:00:00
#SBATCH --ntasks-per-node=15
#SBATCH --output=negspiky_lfsr.log
#SBATCH --partition=stat,hns,normal

# load modules 
ml load julia/1.6.2

# execute script
julia -t 15 simulation_ebci.jl lfsr negspiky
