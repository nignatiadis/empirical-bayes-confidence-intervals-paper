using Pkg
Pkg.activate(".")

using Empirikos

using SpecialExponentialFamilies
using JLD2
using Random


prior = MixtureModel([ Normal(-0.25,.25), Normal(0,1.0)],[0.8, 0.2])



n = 5_000

marginal = marginalize(StandardNormalSample(), prior)





target = Empirikos.PosteriorProbability.(StandardNormalSample.(2.0), Interval(0,nothing))

nreps = 400
ci_list = Vector{Any}(undef, nreps)

rnglock = ReentrantLock()
rng = Random.MersenneTwister(1)

Threads.@threads for i in Base.OneTo(nreps)
    @show i, Threads.threadid()

    lock(rnglock)
    Zs = summarize(discretize(StandardNormalSample.(rand(rng, marginal, n))))
    unlock(rnglock)

    ci_list[i] = Dict{Int, Any}()

    for df=2:13
        @show df
        ef = ExponentialFamily(basemeasure = Uniform(-4.0,4.0), df = df)
        ef_method = SpecialExponentialFamilies.PenalizedMLE(ef=ef; c0=0.001)
        ci_list[i][df] = try confint(ef_method, target, Zs ; level=0.95) catch err; err end
    end
end


JLD2.@save "simulation_expfamily.jld2" ci_list
