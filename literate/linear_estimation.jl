# # Minimax estimation of linear functionals

# This tutorial seeks to illustrate Stein's one-dimensional subfamily heuristic for
# finding minimax optimal linear estimators.

# ## Preliminaries

# Let us load some required packages first 
using MinimaxCalibratedEBayes
using MosekTools
using Setfield
using LaTeXStrings
using LaTeXTabulars

const MCEB = MinimaxCalibratedEBayes

# Some further preliminaries for the plotting setting:
using Plots
using PGFPlotsX
pgfplotsx()
def_size = (850,440)
third_col = RGB(68/255, 69/255, 145/255)
import Plots:pgfx_sanitize_string
pgfx_sanitize_string(s::AbstractString) = s

# # Recap of goal in this section
# Here we seek to demonstrate how we can solve optimization problems of the following form:
# Given a partition of the real line $(-\infty, \infty)$ into bins $I_{k}$, find 
# the constant $Q_0$ and function $Q(\cdot)$ which optimize the following
# optimization problem:
#
# $$\min_{Q_0,Q} \left\{ \sup_{ G \in \mathcal{G}} \{ (Q_0 + E_{G}[Q(X_i)] - L(G))^2\} + \frac{1}{n}\int Q^2(x) \bar{f}(x) dx \right\}$$
# We solve this over all functions $Q(\cdot)$ that are piecewise constant on $I_k$. This 
# corresponds to finding the linear estimator:
#
# $$\hat{L} = Q_0 + \frac{1}{n} \sum_{i=1}^n Q(X_i)$$
#
# that solves a worst-case bias, variance problem with respect
# to the following problem parameters.
# * The convex class of priors (effect size distributions) $\mathcal{G}$.
# * The linear functional$L(G)$ operating on $G \in \mathcal{G}$.
# * A pilot estimator for the marginal density of the observations $\bar{f}(\cdot)$.

# # Main plots of quasi-minimax linear estimators
# Here we reproduce Figures 3,4,5, of the paper and also
# show how the package can be used to derive quasi-minimax linear estimators. 

# ## Marginal Density Target, Gaussian Mixture prior

# We start by defining the prior (effect size) distribution we will use throughout.
# This is used as the ground truth (and would typically not be available) in a real
# analysis. 

prior_dbn = MixtureModel(Normal, [(-2,0.2), (2, 0.2)])

# Note that under this prior $G$, the true marginal density is the following:
marginalize(prior_dbn, StandardNormalSample(0.0))

# Next we define the discretization of the output domain by using the
# `DiscretizedStandardNormalSamples` type.
marginal_grid = -6:0.01:6
Zs_discr = DiscretizedStandardNormalSamples(marginal_grid);

# We populate the `var_proxy` field of `Zs_discr` based on the ground truth.
# That is, for $k$ and bin $I_k$ in the discretization, we compute 
# $$\bar{\nu}(k) = \int_{I_k} \bar{f}(x) dx$$

Zs_discr_var = @set Zs_discr.var_proxy = pdf(marginalize(prior_dbn, Zs_discr));

# We define the linear functional we want to design a linear estimator for,
# namely the marginal density at 0.0, i.e. $f_{G}(0)$.
marginal_target = MarginalDensityTarget(StandardNormalSample(0.0))

# Furthermore we will do this based on 10000 samples.
n_marginal = 10_000

# Finally we define $\mathcal{G}$, the class of effect size distributions we consider:
gmix = GaussianMixturePriorClass(0.2, -3:0.01:3, Mosek.Optimizer, ["QUIET" =>true])

# Finally we are ready to compute the minimax estimator; the `MCEB.RMSE(n_marginal, 0.0)`
# option denotes that we want to optimize the mean-squared error trade-off based
# on the sample size `n_marginal`.

marginal_fit = SteinMinimaxEstimator(Zs_discr_var, gmix,
                                     marginal_target, MCEB.RMSE(n_marginal, 0.0));
									 
# Statistical performance is often improved by adding constraints on the marginal
# density of the observations, i.e., when it is known that $\sup_x |f(x) - \bar{f}(x)| \leq c_m$
# for some $c_m$. This is possible to do in a data-driven way. Here we illustrate by using
# $c_m = 0.02$ around the ground truth density.

C∞ = 0.02
Zs_discr_nbhood = set_neighborhood(Zs_discr, prior_dbn; C∞ = C∞)

# Finally we recompute the minimax estimator under the additional localization constraint:
marginal_fit_nbhood = SteinMinimaxEstimator(Zs_discr_nbhood, gmix,
									 marginal_target, MCEB.RMSE(n_marginal, 0.0));

# We have computed two linear estimators for the marginal density target. A baseline to compare
# against is the Butucea-Comte estimator:

bc_marginal = MCEB.ButuceaComteEstimator(marginal_target;  n=n_marginal);

# Finally, we are ready to plot all the linear estimators, as well as the worst case
# densities. This is Figure 3 in the manuscript.



marginal_density_affine = steinminimaxplot(marginal_fit, marginal_fit_nbhood; size=def_size,
									                   ylim_relative_offset=1.4)
plot!(marginal_density_affine[1], marginal_grid, bc_marginal.(marginal_grid), 
      linestyle=:dot, linecolor = third_col, label="Butucea-Comte")
	  
#md plot(marginal_density_affine)

#jl savefig(marginal_density_affine, "marginal_density_affine.pdf")


# ## Prior Tail Probability, Gaussian Mixture prior
# Next we compute the quasi-minimax estimators in the same setting as above, with the following changes:
# The target we consider is instead the prior tail probability $\mathbb P_G[\mu_i \geq 0]$:

prior_cdf_target = PriorTailProbability(0.0)

# The sample size we consider is
n_prior_cdf = 200

# We compute the minimax estimators over the whole class $\mathcal{G}$, as well as the 
# localized class $\mathcal{G}_m$.

prior_cdf_fit = SteinMinimaxEstimator(Zs_discr_var, gmix,
                                      prior_cdf_target, MCEB.RMSE(n_prior_cdf, 0.0));

prior_cdf_fit_nbhood = SteinMinimaxEstimator(Zs_discr_nbhood, gmix,
									  prior_cdf_target, MCEB.RMSE(n_prior_cdf, 0.0));


# Let us plot the results									  

prior_cdf_affine = steinminimaxplot(prior_cdf_fit, prior_cdf_fit_nbhood; size=def_size,
									  					 ylim_relative_offset=1.4)

#md plot(prior_cdf_affine)
#jl savefig(prior_cdf_affine, "prior_cdf_affine.pdf") 

# ## Prior Density Target, Sobolev prior class

# As our next example we will keep the same sample size as above.
n_prior_density = 200
# However, the target will instead be the density at $0$, i.e., $L(G) = g(0)$:
prior_density_target = PriorDensityTarget(0.0)

# Furthermore, we will now take $\mathcal{G}$ to be the Sobolev class: 
hmclass = MCEB.HermitePriorClass(qmax=90, sobolev_order=2, sobolev_bound=0.5,
                                 solver=Mosek.Optimizer, solver_params = ["QUIET" => true])

# Finally, we will also use a different ground truth $G$
prior_dbn_normal = Normal(0, 2)

# Let us compute again the marginal (discretized) distribution implies by prior_dbn_normal
marginalize(prior_dbn_normal, StandardNormalSample(0.0))
Zs_discr_var_normal = @set Zs_discr.var_proxy = pdf(marginalize(prior_dbn_normal, Zs_discr));
Zs_discr_nbhood_normal = set_neighborhood(Zs_discr, prior_dbn_normal; C∞ = C∞);

# We now compute the minimxa linear estimators (without and with localization):

prior_density_fit = SteinMinimaxEstimator(Zs_discr_var_normal, hmclass,
								          prior_density_target, MCEB.RMSE(n_prior_density,0.0))

prior_density_fit_nbhood = SteinMinimaxEstimator(Zs_discr_nbhood_normal, hmclass,
								   prior_density_target, MCEB.RMSE(n_prior_density,0.0))

# Let us also instantiate the Butucea-Comte estimator
bc_prior = MCEB.ButuceaComteEstimator(prior_density_target;  n=n_prior_density)

# Again we plot the results:

prior_density_affine = steinminimaxplot(prior_density_fit, prior_density_fit_nbhood; size=def_size,
									     ylim_relative_offset=1.3, ylim_panel_e= (0,0.33))

plot!(prior_density_affine[1], marginal_grid, bc_prior.(marginal_grid), 
      linestyle=:dot, linecolor = third_col, label="Butucea-Comte")

#jl savefig(prior_density_affine, "prior_density_affine.pdf")


# ## Tables






Zs_big = DiscretizedStandardNormalSamples(-15:0.01:15)
Zs_big_var = @set Zs_big.var_proxy = pdf(marginalize(prior_dbn, Zs_big))
Zs_big_nbhood = set_neighborhood(Zs_big, prior_dbn; C∞ = C∞)

bc_marginal_affine = DiscretizedAffineEstimator(Zs_big_var.mhist, bc_marginal)



function latex_minimax_tbl(tbl_name, estimator_sets, prior_class, target, n; rounding_digits=4)
	line1=["", L"\se[G]{\hat{L}}",L"\sup_{G \in \mathcal{G}}\lvert\Bias[G]{\hat{L},L}\rvert", L"\sup_{G \in \mathcal{G}_m}\lvert\Bias[G]{\hat{L},L}\rvert"]
	lines= [line1, Rule(),]
	for (est_name, estimator_set) in estimator_sets
		estimator = estimator_set[1]
		Zs = estimator_set[2]
		Zs_nbhood = estimator_set[3]
		bias_nonbhood = worst_case_bias(estimator, Zs, prior_class, target).max_abs_bias
		bias_nbhood = worst_case_bias(estimator, Zs_nbhood, prior_class, target).max_abs_bias
		se_calc = MCEB.std_proxy(estimator, Zs, n)
		se_calc, bias_nonbhood, bias_nbhood = round.( (se_calc, bias_nonbhood, bias_nbhood), digits=rounding_digits)
		push!(lines, [est_name, se_calc, bias_nonbhood, bias_nbhood])
	end
	latex_tabular(tbl_name, Tabular("l|ccc"), lines) 
end 



latex_minimax_tbl("marginal_density_affine.tex",
                  ["Butucea-Comte" => (bc_marginal_affine, Zs_big_var, Zs_big_nbhood),
				   "Minimax" => (marginal_fit.Q,Zs_discr_var, Zs_discr_nbhood),
                   L"Minimax-$\infty$" => (marginal_fit_nbhood.Q, Zs_discr_var, Zs_discr_nbhood)],  
                   gmix, marginal_target, n_marginal)




latex_minimax_tbl("prior_cdf_affine.tex",
                ["Minimax" => (prior_cdf_fit.Q,Zs_discr_var, Zs_discr_nbhood),
                 L"Minimax-$\infty$" => (prior_cdf_fit_nbhood.Q, Zs_discr_var, Zs_discr_nbhood)],  
                 gmix, prior_cdf_target, n_prior_cdf)


Zs_big_var_normal = @set Zs_big.var_proxy = pdf(marginalize(prior_dbn_normal, Zs_big))
Zs_big_nbhood_normal = set_neighborhood(Zs_big, prior_dbn_normal; C∞ = C∞)


bc_prior_affine = DiscretizedAffineEstimator(Zs_big_var_normal.mhist, bc_prior)

latex_minimax_tbl("prior_density_affine.tex",
				  ["Butucea-Comte" => (bc_prior_affine, Zs_big_var_normal, Zs_big_nbhood_normal),
				   "Minimax" => (prior_density_fit.Q,Zs_discr_var_normal, Zs_discr_nbhood_normal),
				   L"Minimax-$\infty$" => (prior_density_fit_nbhood.Q, Zs_discr_var_normal, Zs_discr_nbhood_normal)],  
				   hmclass, prior_density_target, n_prior_density)


