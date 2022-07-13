using Pkg
Pkg.activate(".")
using Empirikos
using Distributions
using Plots
using SpecialFunctions
using MosekTools
using Random
using LaTeXStrings
using PGFPlotsX
using JuMP
using Dictionaries


begin
    pgfplotsx()
    empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
end

theme(
    :default;
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    grid = nothing,
    frame = :box,
    legendfonthalign = :left,
    thickness_scaling = 1.3,
    size = (420, 330),
)

#----------------------
# Figure 2, panel a)
#----------------------

## Some sanity checks and helper code

a = 2.0
tmp_gamma = Gamma(a, 0.1)

### Helper functions to work with density of log(U), with U~Gamma
function logrv_variance(gamma::Gamma)
    trigamma(gamma.α)
end

function logrv_expectation(gamma::Gamma)
    digamma(gamma.α) + log(gamma.θ)
end

function logrv_density(gamma::Gamma, t)
    exp(t)*pdf(gamma, exp(t))
end


### Check that claims for variance of log(components) of Gamma(2, U) is correct

### Analytical formula for variance
logrv_variance(tmp_gamma)

### Sanity check through Monte Carlo Simulation
var(log.(rand(tmp_gamma, 1_000_000)))



## Start plotting the densities



gammas_for_plot = Gamma.(a, 10.0 .^ (-2:1:0))
ts_for_plot = -7:0.02:3

fs = [logrv_density(gamma, t) for t in ts_for_plot, gamma in gammas_for_plot]

### Sanity check for their expectations:
logrv_expectation.(gammas_for_plot)

component_densities_plot = plot(
    ts_for_plot,
    fs,
    color = [:purple :darkblue :teal],
    label = [L"\theta=0.01" L"\theta=0.1" L"\theta=1"],
    linestyle = [:dot :dash :solid],
    legend = :topleft,
    xlabel = L"\log(\mu)",
    xticks = -6:2:2,
    xlim = (-8 , 4),
    ylabel = "Density",
    size = (380, 280)
)

savefig(component_densities_plot, "component_densities_plot.tikz")

#-----------------------------------------------------------------------
#  Code for inference (Figure 2b,c)  Prior density, Posterior density
#-----------------------------------------------------------------------

## Load data
Zs = Empirikos.Bichsel.ebayes_samples()


## Define Bayes Targets
ts_density = -7:0.2:3
prior_density_targets = Empirikos.PriorDensity.(exp.(ts_density)) .* exp.(ts_density)
posterior_density_targets_1 = Empirikos.PosteriorDensity.(PoissonSample(1), exp.(ts_density)) .* exp.(ts_density)
posterior_density_targets_3 = Empirikos.PosteriorDensity.(PoissonSample(3), exp.(ts_density)) .* exp.(ts_density)

## Load solver
quiet_mosek = optimizer_with_attributes(Mosek.Optimizer, "QUIET" => true)


## Define smooth class of priors
gamma_mix = Empirikos.MixturePriorClass(Gamma.(a, 10.0 .^ (-2:0.01:0)))

chisq_floc = Empirikos.ChiSquaredFLocalization(0.05)

floc_chisq_smooth = FLocalizationInterval(flocalization = chisq_floc,
    convexclass= gamma_mix, solver=quiet_mosek)

amari_chisq_smooth = AMARI(
    flocalization = fit(Empirikos.ChiSquaredFLocalization(0.01), Zs),
    convexclass=gamma_mix,
    solver=quiet_mosek,
    discretizer=integer_discretizer(0:5),
    plugin_G=fit(Empirikos.KolmogorovSmirnovMinimumDistance(gamma_mix, quiet_mosek), Zs))

#--------------------------------------------
## Conduct inference for prior density
#--------------------------------------------

floc_chisq_smooth_cis_density = confint.(floc_chisq_smooth, prior_density_targets, Zs)
amari_chisq_smooth_cis_density = confint.(amari_chisq_smooth, prior_density_targets, Zs)


### Sanity check:
sum(getproperty.(floc_chisq_smooth_cis_density, :estimate) .* 0.2) # should be approx 1

## Figure 2b

prior_density_plot = begin
	prior_density_plot = plot(ts_density, floc_chisq_smooth_cis_density,
		label=L"\chi^2\textrm{-F-Loc}", show_ribbon=false, alpha=0.9, color=:black)
	plot!(prior_density_plot, ts_density, amari_chisq_smooth_cis_density,
		label="AMARI",show_ribbon=true, fillcolor=:blue, fillalpha=0.4,
        xlabel = L"\log(\mu)", ylabel="Density",
        xticks = -6:2:2,
        xlim = (-8 , 4),
        size = (380, 280))
end

savefig(prior_density_plot, "cis_prior_density_bichsel.tikz" )

#--------------------------------------------
# Conduct inference for posterior density
#--------------------------------------------

floc_chisq_smooth_cis_posterior_1 = confint.(floc_chisq_smooth, posterior_density_targets_1, Zs)
floc_chisq_smooth_cis_posterior_3 = confint.(floc_chisq_smooth, posterior_density_targets_3, Zs)

amari_chisq_smooth_cis_posterior_1 = confint.(amari_chisq_smooth, posterior_density_targets_1, Zs)
amari_chisq_smooth_cis_posterior_3 = confint.(amari_chisq_smooth, posterior_density_targets_3, Zs)

### Sanity check:
sum(getproperty.(floc_chisq_smooth_cis_posterior_1, :estimate) .* 0.2) # should be approx 1
sum(getproperty.(floc_chisq_smooth_cis_posterior_3, :estimate) .* 0.2) # should be approx 1


## Figure 2c

posterior_density_plot = begin
	posterior_density_plot = plot(ts_density, floc_chisq_smooth_cis_posterior_1,
		label=L"\chi^2\textrm{-F-Loc } \mid \; z=1", show_ribbon=false, alpha=0.9, color=:black)
	plot!(posterior_density_plot, ts_density, amari_chisq_smooth_cis_posterior_1,
		label=L"\textrm{AMARI }\;\, \mid \; z=1",show_ribbon=true, fillcolor=:blue, fillalpha=0.4,
        xlabel = L"\log(\mu)", ylabel=L"\textrm{Posterior Density} \mid Z=z",
        xticks = -6:2:2,
        xlim = (-8 , 4),
        size = (380, 280))
    plot!(posterior_density_plot, ts_density, floc_chisq_smooth_cis_posterior_3,
		label=L"\chi^2\textrm{-F-Loc } \mid \; z=3", linestyle=:dot, show_ribbon=false,
        alpha=1, linecolor=:brown)
    plot!(posterior_density_plot, ts_density, amari_chisq_smooth_cis_posterior_3,
		label=L"\textrm{AMARI }\;\, \mid \; z=3",show_ribbon=true, fillcolor=:darkorange, fillalpha=0.6)
end

savefig(posterior_density_plot, "posterior_density_plot.tikz")

#--------------------------------------------------------------------
# Inference for the posterior mean (Table 1)
#--------------------------------------------------------------------

postmean_targets = PosteriorMean.(PoissonSample.(0:4))

#--------------------------------------------------------------------
# AMARI with smooth density class
#--------------------------------------------------------------------

amari_chisq_smooth_cis_postmean = confint.(amari_chisq_smooth, postmean_targets, Zs)

#--------------------------------------------------------------------
# Karlis. Tzougas, and Frangos Bootstrap
#--------------------------------------------------------------------

# Remark: For previous approaches (as described in manuscript),
# we explicitly merge all counts >= 5 and treat them as one category.
# However, the raw data contains the counts for Z=5 and Z=6 separately, so we
# use that information as well for the remaining approaches.
Zs_full = Empirikos.Bichsel.ebayes_samples(;combine=false)

### Sanity check:
nobs(Zs_full) == nobs(Zs)

𝒢 = DiscretePriorClass(0.0:0.01:5.0)

B = 5_000 # number of Bootstrap resamples

# Initial fit of NPMLE
npmle = Empirikos.NPMLE(convexclass = 𝒢, solver = quiet_mosek)
npmle_fit = fit(npmle, Zs_full)


boot_npmle_fits = Vector{Any}(undef, B)
Random.seed!(1)
for b in 1:B
    μs_resample = rand(npmle_fit.prior,  nobs(Zs_full))
    Zs_resample = PoissonSample.(rand.( Poisson.(μs_resample)))
    Zs_resample_summary = Empirikos.summarize(Zs_resample)
    boot_npmle_fits[b] = fit(npmle, Zs_resample_summary).prior
end

function perc_ci(target, boot_priors)
    boot_stats = [target(G) for G in boot_priors]
    _cis = quantile(boot_stats, (0.025, 0.975))
    (target=target, lower = _cis[1], upper = _cis[2])
end

cis_karlis_tzougas_frangos = perc_ci.(postmean_targets, Ref(boot_npmle_fits))

#----------------------------------
# Pensky's proposal:
#----------------------------------

bichsel_dict = Dictionary(Zs_full.store)
bichsel_freqs = bichsel_dict ./ sum(bichsel_dict)

function pensky_ci(z, dict)
    n = sum(dict)
    freqs = dict ./ n
    freq_z = freqs[PoissonSample(z)]
    freq_zp1 = freqs[PoissonSample(z+1)]
    hat_theta = (z+1)*freq_zp1/freq_z

    κ = quantile(Normal(), 0.975)
    pm = 2κ*hat_theta/(sqrt(n)) * ( sqrt(freq_zp1 * (1-freq_zp1))/freq_zp1 +
                                   sqrt(freq_z * (1-freq_z))/freq_z )
    (hat_theta = hat_theta, lower_ci = hat_theta - pm, upper_ci = hat_theta + pm)
end

pensky_ci.(0:4, Ref(bichsel_dict))




#--------------------------------------------------------------------
# Inference for the posterior variance (Table 2)
#--------------------------------------------------------------------

postvariance_targets= Empirikos.PosteriorVariance.(PoissonSample.(0:4))

#-------------------------
## Smooth F-Localization
#-------------------------

floc_chisq_smooth_cis_postvariance = confint.(floc_chisq_smooth, postvariance_targets, Zs)


### Sanity check: The below should be ≈ to the PosteriorVariance interval given Z=0
sanity_smooth_var = confint(floc_chisq_smooth, Empirikos.PosteriorSecondMoment(PoissonSample(0), 0.135), Zs)

# Transform to report posterior standard deviation

function square_root_ci_transform(ci::Empirikos.ConfidenceInterval)
    ci = Empirikos.LowerUpperConfidenceInterval(lower = sqrt(ci.lower),
            upper = sqrt(ci.upper),
            estimate = sqrt(ci.estimate),
            α = ci.α)
end

# CIs for posterior standard deviation
square_root_ci_transform.(floc_chisq_smooth_cis_postvariance)

#-------------------------
## F-Localization with 𝒢
#-------------------------


floc_method_chisq = FLocalizationInterval(flocalization = chisq_floc,
                                       convexclass= 𝒢, solver=quiet_mosek)

floc_chisq_cis_postvariance = confint.(floc_method_chisq, postvariance_targets, Zs)

### Sanity check: The below should be ≈ to the PosteriorVariance interval given Z=0
sanity_var = confint(floc_method_chisq, Empirikos.PosteriorSecondMoment(PoissonSample(0), 0.135), Zs)


# CIs for posterior standard deviation
square_root_ci_transform.(floc_chisq_cis_postvariance)

#----------------------------------
# AMARI with discrete gcal P(0,5)
#----------------------------------


amari_chisq = AMARI(
    flocalization = fit(Empirikos.ChiSquaredFLocalization(0.01), Zs),
    solver=quiet_mosek, convexclass=𝒢, discretizer=integer_discretizer(0:5),
    plugin_G=fit(Empirikos.KolmogorovSmirnovMinimumDistance(𝒢, quiet_mosek),Zs))


confint(amari_chisq, PosteriorMean(PoissonSample(1)), Zs)

function amari_for_posterior_variance(amari::AMARI, target::Empirikos.PosteriorVariance, Zs)
    post_mean = PosteriorMean(location(target))
    post_mean_hat = post_mean(amari.plugin_G.prior)

    post_second_moment_m = Empirikos.PosteriorSecondMoment(location(target), post_mean_hat)
    ci_upper = confint(amari, post_second_moment_m, Zs; level=0.975, tail=:left)

    floc_worst_case = FLocalizationInterval(flocalization = amari.flocalization,
        convexclass = amari.convexclass,
        solver= amari.solver)

    cis_first_moment = confint(floc_worst_case, post_mean, Zs)
    post_mean_grid = range(cis_first_moment.lower, cis_first_moment.upper; length=20)
    post_second_moment_grid = Empirikos.PosteriorSecondMoment.(location(target), post_mean_grid)

    ci_lower_grid = confint.(amari, post_second_moment_grid, Zs; level=0.975, tail=:right)
    ci_lower = minimum(getproperty.(ci_lower_grid, :lower))
    Empirikos.LowerUpperConfidenceInterval(target=target, lower=ci_lower , upper=ci_upper.upper, α=0.05)
end

amari_chisq_postvariance = [amari_for_posterior_variance(amari_chisq, target, Zs) for target in postvariance_targets]


square_root_ci_transform.(amari_chisq_postvariance)

#----------------------------------
# AMARI with smooth density class
#----------------------------------

amari_chisq_smooth_postvariance = [amari_for_posterior_variance(amari_chisq_smooth, target, Zs) for target in postvariance_targets]


square_root_ci_transform.(amari_chisq_smooth_postvariance)
