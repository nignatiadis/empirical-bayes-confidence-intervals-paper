# empirical-bayes-confidence-intervals-paper

A repository for reproducing the numerical results in the following preprint:

  >Ignatiadis, Nikolaos, and Stefan Wager. "Bias-Aware Confidence Intervals for Empirical Bayes Analysis." [arXiv:1902.02774](https://arxiv.org/abs/1902.02774) (2019)

See [Empirikos.jl](https://github.com/nignatiadis/Empirikos.jl) for the Julia package implementing the proposed method.

All computations were run on [Julia](https://julialang.org/) version 1.6.1 and [Mosek](https://www.mosek.com/) version 9.2. See `Manifest.toml` for specifications of all Julia packages used.

## File description

* **Setup:** `Manifest.toml` and `Project.toml`, specifications of Julia packages used.
* **Real data examples:** `data_lord_cressie.jl`, `data_prostate.jl` and `data_bichsel.jl` reproduce the three real data examples of the paper (Figure 1, Figure2 and Table 2). These examples have been implemented as [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebooks.
* **Simulations:**
  * `simulation_ebci.jl`: Main script for running simulations. It can be called for example as `julia -t 10 simulation_ebci.jl postmean negspiky` where 10 is the number of threads. See the `_simulation_*.sbatch` files for all the calls of this script reported in the manuscript.
  * `simulation_expfamily.jl`: Script to run simulations with Logspline and varying degrees of freedom.
  * `simulation_plots.jl`: Takes the output of the simulation scripts above and generates Figures 3-7 of the paper .
* **Other:**
  * `asymptotic_ci_length.jl`
## Repository history

* Version 2 of this manuscript on arXiv ([arXiv:1902.02774v2](https://arxiv.org/abs/1902.02774v2)) was substantially different. See this repository at the [arXiv_V2 release tag](https://github.com/nignatiadis/empirical-bayes-confidence-intervals-paper/releases/tag/arXiv_v2) to reproduce the results of that version.
* Similarly, you can find the code for version 3  ([arXiv:1902.02774v3](https://arxiv.org/abs/1902.02774v3)) at the [[arXiv_V3 release tag](https://github.com/nignatiadis/empirical-bayes-confidence-intervals-paper/releases/tag/arXiv_v3) ].