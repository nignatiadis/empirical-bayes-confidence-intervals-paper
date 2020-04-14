# MCEBPaper

A repository for reproducing the numerical results in the following preprint:

  >Ignatiadis, Nikolaos, and Stefan Wager. "Bias-Aware Confidence Intervals for Empirical Bayes Analysis." [arXiv:1902.02774](https://arxiv.org/abs/1902.02774) (2019)

Also see [MinimaxCalibratedEBayes.jl](https://github.com/nignatiadis/MinimaxCalibratedEBayes.jl) for the Julia package implementing the proposed method. 

All computations were run on [Julia](https://julialang.org/) version 1.3.1 and [Mosek](https://www.mosek.com/) version 9.2. See `Manifest.toml` for specifications of Julia packages used. 

The folder structure is as follows (cf. [DrWatson.jl](https://github.com/JuliaDynamics/DrWatson.jl)):

* **src:** Helper functions that are used throughout this repository. `mceb_sim_eval.jl` contains a helper function to run simulations, while `plotting_helpers.jl` contains some functions that assist
with plotting.
* **scripts:** These are the scripts that run the simulations, i.e., they instantiate all simulation 
parameters, run then Monte Carlo replicates of the experiment and finally save the data. Data is stored in the `data/sims` folder. 
* **data/sims:** Folder in which output from the scripts in `scripts` folder is saved.
* **literate:** These files generate the figures and tables in the paper. They can be either standalone or use results from the `data/sim` folder.


