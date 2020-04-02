#!/bin/bash
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 7
#SBATCH --time 20:00:00
#SBATCH --partition=hns

ml julia/1.3.1
julia ./unimodal_posterior_mean.jl