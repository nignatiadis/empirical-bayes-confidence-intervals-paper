using Pkg
Pkg.activate(".")

using Empirikos

using MosekTools
using JuMP
using Setfield
using SpecialExponentialFamilies
using JLD2
using Random

@show target_string = ARGS[1]
@show prior_string = ARGS[2]

quiet_mosek = optimizer_with_attributes(Mosek.Optimizer,  "QUIET" => true,
                                        "MSK_IPAR_NUM_THREADS" => 1)


if prior_string == "spiky"
    prior = Empirikos.AshPriors[:Spiky]
elseif prior_string == "negspiky"
    prior = MixtureModel([ Normal(-0.25,.25), Normal(0,1.0)],[0.8, 0.2])
else
    throw("only spiky and negspiky implemented")
end


n = 5_000

marginal = marginalize(StandardNormalSample(), prior)


a_min = -4.0
a_max = +4.0


gcal_scalemix = Empirikos.set_defaults(GaussianScaleMixtureClass(), StandardNormalSample.([-1.0;1.0]); hints = Dict(:grid_scaling => 1.1, :σ_max => 15.0))
gcal_locmix = MixturePriorClass(Normal.(-4:0.075:4, 0.25))

discr = Empirikos.Discretizer(a_min:0.01:a_max)

dkw_nbhood = DvoretzkyKieferWolfowitz(0.05)
infty_nbhood = Empirikos.InfinityNormDensityBand(a_min=a_min, a_max=a_max);






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



ef = ExponentialFamily(basemeasure = Uniform(-4.0,4.0), df = 5)

ef_method = SpecialExponentialFamilies.PenalizedMLE(ef=ef; c0=0.001)


ts= -3:0.2:3

if target_string == "postmean"
    targets = Empirikos.PosteriorMean.(StandardNormalSample.(ts))
elseif target_string == "lfsr"
    targets = Empirikos.PosteriorProbability.(StandardNormalSample.(ts), Interval(0,nothing))
else
    throw("Only lfsr or posterior mean allowed")
end

# Posterior Mean
nreps = 400
ci_list = Vector{Any}(undef, nreps)

rnglock = ReentrantLock()
rng = Random.MersenneTwister(1)

Threads.@threads for i in Base.OneTo(nreps)
    @show i, Threads.threadid()

    lock(rnglock)
    Zs = StandardNormalSample.(rand(rng, marginal, n))
    unlock(rnglock)

    ci_dkw_scalemix = try confint.(nbhood_method_dkw_scalemix, targets, Zs) catch err; err end
    ci_kde_scalemix = try confint.(nbhood_method_kde_scalemix, targets, Zs) catch err; err end
    ci_lam_scalemix = try confint.(lam_kde_scalemix, targets, Zs) catch err; err end

    ci_dkw_locmix = try confint.(nbhood_method_dkw_locmix, targets, Zs) catch err; err end
    ci_kde_locmix = try confint.(nbhood_method_kde_locmix, targets, Zs) catch err; err end
    ci_lam_locmix = try confint.(lam_kde_locmix, targets, Zs) catch err; err end

    ci_logspline = try confint.(ef_method, targets, summarize(discretize(Zs)); level=0.95) catch err; err end

    ci_list[i] = (dkw_scalemix = ci_dkw_scalemix,
                  kde_scalemix = ci_kde_scalemix,
                  lam_scalemix = ci_lam_scalemix,
                  dkw_locmix = ci_dkw_locmix,
                  kde_locmix = ci_kde_locmix,
                  lam_locmix = ci_lam_locmix,
                  logspline = ci_logspline,
                  id =  Threads.threadid())
end


JLD2.@save "$(prior_string)_$(target_string).jld2" ci_list
