# empirical-bayes-confidence-intervals-rejoinder

A repository for reproducing the numerical results in the rejoinder:

  >Ignatiadis, Nikolaos, and Stefan Wager. "Rejoinder: Confidence Intervals for Nonparametric Empirical Bayes Analysis." Journal of the American Statistical Association: Theory and Methods (2022)


## File description

* **Setup:** `Manifest.toml` and `Project.toml`, specifications of versions of Julia packages used.
* **Main reproduction code:** 
  * `bias_aware_plots.jl`: Code to reproduce Figure 1 of the rejoinder.
  * `bichsel_smooth.jl`: Script to reproduce Figure 2, and Tables 1-2 of the rejoinder.

## Requirements and version-info for reproduction

* The [Mosek](https://www.mosek.com/) convex programming solver, version 9.2, was used. Mosek requires a license (there is an option for a free academic license).
* [Julia](https://julialang.org/) version 1.7.2.
* All required Julia packages and their versions are specified in `Project.toml` and `Manifest.toml`. They may be installed automatically by starting a Julia session in this folder and typing:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```
