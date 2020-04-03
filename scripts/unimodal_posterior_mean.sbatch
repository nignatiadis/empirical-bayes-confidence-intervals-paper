#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 9
#SBATCH --time 10:00:00
#SBATCH --partition=hns

ml julia/1.3.1
julia ./unimodal_posterior_mean.jl