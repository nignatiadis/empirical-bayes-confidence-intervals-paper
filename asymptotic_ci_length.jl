using Empirikos
using Random
using Plots
using PGFPlotsX
using MosekTools
using LaTeXStrings
using JuMP
using JLD2

pgfplotsx()


discr = integer_discretizer(0:6)

K = 3
density_target = MarginalDensity(PoissonSample(K))
postmean_target = PosteriorMean(PoissonSample(K))

true_G = Uniform(0,2)

𝒢 = DiscretePriorClass(0:0.01:4)

# analytic expressions for marginal density
f2 = 1/2 - 5/(2*exp(2))
f3 = (1-19/(3*exp(2)))/2
f4 = 1/2 - 7/(2*exp(2))

amari_dkw = AMARI(
    flocalization = Empirikos.DvoretzkyKieferWolfowitz(0.01),
    solver=Mosek.Optimizer, convexclass=𝒢, discretizer=discr,
    modulus_model=Empirikos.ModulusModelWithF)

floc_method = FLocalizationInterval(flocalization= Empirikos.DvoretzkyKieferWolfowitz(0.05),
    solver=Mosek.Optimizer,
    convexclass=𝒢)


n = Int.(1e8)
μs = rand(true_G, n)
Zs = rand.(Poisson.(μs))


function evaluate_at_n(true_G, amari, floc, target, n; ci_only=false)
    μs = rand(true_G, n)
    Zs = rand.(Poisson.(μs))
    Zs_discr = summarize(discr.(Zs))
    Zs_summary = Empirikos.MultinomialSummary(PoissonSample.(keys(Zs_discr)), values(Zs_discr))

    if ci_only
        floc_fit = nothing
        amari_fit = nothing
        floc_ci = confint(floc, target, Zs_summary; level=0.95)
        amari_ci = confint(amari, target, Zs_summary; level=0.95)
    else
        floc_fit = fit(floc, target, Zs_summary)
        floc_ci = confint(floc_fit; level=0.95)
        amari_fit = fit(amari, target, Zs_summary)
        amari_ci = confint(amari_fit, target, Zs_summary; level=0.95)
    end
    (n=n, amari_fit=amari_fit, amari_ci=amari_ci, floc_fit=floc_fit, floc_ci=floc_ci)
end


Random.seed!(1)

nreps = 50
ns = floor.(Int64, 10 .^ (2:0.5:8))
results_density = Matrix{Any}(undef, length(ns), nreps)


for (i,n) in enumerate(ns)
    for j in Base.OneTo( nreps)
        ci_only = j > 1
        results_density[i,j] = evaluate_at_n(true_G, amari_dkw,
                        floc_method, density_target, n; ci_only=ci_only)
    end
end

#jldsave("results_density.jld2"; results_density)
#results_density = load("results_density.jld2", "results_density")


α = 0.05
amari_ci_lengths = mean([res.amari_ci.upper - res.amari_ci.lower for res in results_density]; dims=2)
dkw_ci_lengths = mean([res.floc_ci.upper - res.floc_ci.lower for res in results_density]; dims=2)
theory_ci_lengths = 2*quantile(Normal(), 1-α/2)*sqrt.(f3*(1-f3)./ns)
theory_dkw_ci_lengths = 2*2*sqrt.(log(2/α) /2 ./ ns)


theme(
    :default;
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    grid = nothing,
    frame = :box,
    thickness_scaling = 1.3,
    legendfonthalign = :left,
    size = (420, 330),
)


poisson_density_relative_efficiency =
    plot(ns, [amari_ci_lengths theory_ci_lengths dkw_ci_lengths theory_dkw_ci_lengths],
       scale=:log10,
       label=[L"\textrm{AMARI}" L"\textrm{AMARI asymp.}" L"\textrm{DKW-F-Loc}" L"\textrm{DKW-F-Loc asymp.}"],
       legend= :topright,
       color=[:blue :blue :darkorange :darkorange],
       linestyle=[:solid :dot :solid :dot],
       xguide = L"n",
       yguide = "Expected CI length",
       size=(420,300)
       )


savefig(poisson_density_relative_efficiency, "poisson_density_relative_efficiency.tikz")


zs = 0:7
Qs_3 = results_density[3,1].amari_fit.Q.(PoissonSample.(zs))
Qs_7 = results_density[7,1].amari_fit.Q.(PoissonSample.(zs))
Qs_11 = results_density[11,1].amari_fit.Q.(PoissonSample.(zs))

_strings = ["Q($z)" for z in zs]

Q_plot = plot(zs .- 0.15, Qs_3, seriestype=:sticks, label = L"n=10^3", color=:lightgrey,
    legend=:topright, linewidth=1.3, yticks=-0.5:0.25:1.0,
    ylims=(-0.55,1.05), xticks=(0:7,_strings), size=(420,300),
    xguide=L"\phantom{-}"
    )

plot!(Q_plot, zs, Qs_7, seriestype=:sticks,  label = L"n=10^5",
     linewidth=1.3, color=:lightblue)
plot!(Q_plot, zs .+ 0.15, Qs_11, seriestype=:sticks,  label = L"n=10^7",
     linewidth=1.3, color=:darkblue, linealpha=0.4)

savefig(Q_plot, "Q_plot.tikz")



Random.seed!(1)

nreps = 50
results_postmean  = Matrix{Any}(undef, length(ns), nreps)

for (i,n) in enumerate(ns)
    for j in Base.OneTo( nreps)
        @show (i,j)
        results_postmean[i, j] = evaluate_at_n(true_G, amari_dkw,
                        floc_method, postmean_target, n; ci_only=true)
    end
end


#jldsave("results_postmean.jld2"; results_postmean)
#results_postmean = load("results_postmean.jld2", "results_postmean")


α = 0.05
postmean_amari_ci_lengths = mean([res.amari_ci.upper - res.amari_ci.lower for res in results_postmean];dims=2)
postmean_dkw_ci_lengths = mean([res.floc_ci.upper - res.floc_ci.lower for res in results_postmean];dims=2)
var_delta = (K+1)^2*f4*(1 + f4/f3)/abs2(f3)
postmean_theory_ci_lengths = 2*quantile(Normal(), 1-α/2)*sqrt(var_delta)./sqrt.(ns)
postmean_theory_dkw_ci_lengths = (K+1)*2*2*sqrt.(log(2/α) /2 ./ ns) * (1 + f4/f3) / f3
true_post_mean = (K+1)*f4/f3


poisson_postmean_relative_efficiency =
    plot(ns, [postmean_amari_ci_lengths postmean_theory_ci_lengths postmean_dkw_ci_lengths postmean_theory_dkw_ci_lengths],
       scale=:log10,
       label=[L"\textrm{AMARI}" L"\textrm{AMARI asymp.}" L"\textrm{DKW-F-Loc}" L"\textrm{DKW-F-Loc asymp.}"],
       legend= :topright,
       color=[:blue :blue :darkorange :darkorange],
       linestyle=[:solid :dot :solid :dot],
       xguide = L"n",
       yguide = "Expected CI length",
       size=(420,300)
       )



savefig(poisson_postmean_relative_efficiency, "poisson_postmean_relative_efficiency.tikz")
