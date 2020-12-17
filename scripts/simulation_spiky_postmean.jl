using Pkg
Pkg.activate(".")

using Empirikos

using MosekTools
using JuMP
using Setfield
using JLD2

quiet_mosek = optimizer_with_attributes(Mosek.Optimizer,  "QUIET" => true,
                                        "MSK_IPAR_NUM_THREADS" => 1)


prior = Empirikos.AshPriors[:Spiky]
n = 10_000

#xs = -4:0.01:4

#plot(xs, pdf.(prior,xs), frame=:box,
#        grid=nothing, color=:purple,
#         xguide=L"\mu",yguide=L"g(\mu)",thickness_scaling=2, label=nothing)


marginal = marginalize(StandardNormalSample(), prior)


a_min = -4.0
a_max = +4.0


gcal_scalemix = Empirikos.set_defaults(GaussianScaleMixtureClass(), StandardNormalSample.([-1.0;1.0]); hints = Dict(:grid_scaling => 1.1, :σ_max => 15.0))
gcal_locmix = MixturePriorClass(Normal.(-4:0.075:4, 0.25))

discr = Empirikos.Discretizer(-a_min:0.01:a_max)

dkw_nbhood = DvoretzkyKieferWolfowitz(0.05)
infty_nbhood = Empirikos.InfinityNormDensityBand(a_min=a_min,a_max=a_max);



nbhood_method_dkw_scalemix = NeighborhoodWorstCase(neighborhood = dkw_nbhood,
                               convexclass = gcal_scalemix, solver = quiet_mosek)

nbhood_method_kde_scalemix = NeighborhoodWorstCase(neighborhood = infty_nbhood,
                               convexclass = gcal_scalemix, solver = quiet_mosek)

lam_kde_scalemix = Empirikos.LocalizedAffineMinimax(neighborhood = (@set infty_nbhood.α=0.01),
                            discretizer=discr,
                            solver=quiet_mosek, convexclass=gcal_scalemix)

nbhood_method_dkw_locmix = NeighborhoodWorstCase(neighborhood = dkw_nbhood,
                               convexclass = gcal_locmix, solver = quiet_mosek)

nbhood_method_kde_locmix = NeighborhoodWorstCase(neighborhood = infty_nbhood,
                               convexclass = gcal_locmix, solver = quiet_mosek)


lam_kde_locmix = Empirikos.LocalizedAffineMinimax(neighborhood = (@set infty_nbhood.α=0.01),
                            discretizer=discr,
                            solver=quiet_mosek, convexclass=gcal_locmix)


ts= -3:0.2:3

targets = Empirikos.PosteriorMean.(StandardNormalSample.(ts))


# Posterior Mean
nreps = 200
ci_list = Vector{Any}(undef, nreps)

Threads.@threads for i in Base.OneTo(nreps)
    @show i, Threads.threadid()
    Zs = StandardNormalSample.(rand(marginal, n))

    ci_dkw_scalemix = confint.(nbhood_method_dkw_scalemix, targets, Zs)
    ci_kde_scalemix = confint.(nbhood_method_kde_scalemix, targets, Zs)
    ci_lam_scalemix = confint.(lam_kde_scalemix, targets, Zs)

    ci_dkw_locmix = confint.(nbhood_method_dkw_locmix, targets, Zs)
    ci_kde_locmix = confint.(nbhood_method_kde_locmix, targets, Zs)
    ci_lam_locmix = confint.(lam_kde_locmix, targets, Zs)

    ci_list[i] = (dkw_scalemix = ci_dkw_scalemix,
                  kde_scalemix = ci_kde_scalemix,
                  lam_scalemix = ci_lam_scalemix,
                  dkw_locmix = ci_dkw_locmix,
                  kde_locmix = ci_kde_locmix,
                  lam_locmix = ci_lam_locmix,
                  id =  Threads.threadid())
end


JLD2.@save "spiky_postmean.jld2" ci_list
