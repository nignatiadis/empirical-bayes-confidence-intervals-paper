using Pkg
Pkg.activate(".")

using JLD2
using Empirikos
using CategoricalArrays
using DataFrames
using LaTeXStrings
using StatsPlots
using SpecialExponentialFamilies
using FileIO
using Test

begin
    pgfplotsx()
    empty!(PGFPlotsX.CUSTOM_PREAMBLE)
    push!(PGFPlotsX.CUSTOM_PREAMBLE, raw"\usepackage{amssymb}")
    push!(
        PGFPlotsX.CUSTOM_PREAMBLE,
        raw"\newcommand{\PP}[2][]{\mathbb{P}_{#1}\left[#2\right]}",
    )
    push!(
        PGFPlotsX.CUSTOM_PREAMBLE,
        raw"\newcommand{\EE}[2][]{\mathbb{E}_{#1}\left[#2\right]}",
    )
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


spiky_prior = Empirikos.AshPriors[:Spiky]
negspiky_prior = MixtureModel([Normal(-0.25, 0.25), Normal(0, 1.0)], [0.8, 0.2])

tts = -4:0.01:4
prior_densities_plot = plot(
    tts,
    pdf.(spiky_prior, tts),
    color = :purple,
    linealpha = 0.8,
    label = "Spiky",
    linestyle = :solid,
    legend = :topright,
    xlabel = L"\mu",
    ylabel = L"dG(\mu)/d\lambda^{\textrm{Leb}}",
    size = (380, 280),
)
plot!(
    prior_densities_plot,
    tts,
    pdf.(negspiky_prior, tts),
    color = :darkblue,
    linealpha = 0.6,
    label = "NegSpiky",
    linestyle = :dash,
)

marginal_densities_plot = plot(
    tts,
    pdf.(spiky_prior, StandardNormalSample.(tts)),
    color = :purple,
    linealpha = 0.8,
    label = "Spiky",
    linestyle = :solid,
    legend = :topright,
    xlabel = L"z",
    ylabel = L"f_G(z)",
    size = (380, 280),
)
plot!(
    marginal_densities_plot,
    tts,
    pdf.(negspiky_prior, StandardNormalSample.(tts)),
    color = :darkblue,
    linealpha = 0.6,
    label = "NegSpiky",
    linestyle = :dash,
)

savefig(prior_densities_plot, "prior_densities.tikz")
savefig(marginal_densities_plot, "marginal_densities.tikz")

negspiky_lfsr = load("negspiky_lfsr.jld2", "ci_list");
spiky_lfsr = load("spiky_lfsr.jld2", "ci_list");
negspiky_postmean = load("negspiky_postmean.jld2", "ci_list");
spiky_postmean = load("spiky_postmean.jld2", "ci_list");


methods_list = [
    :dkw_scalemix
    :kde_scalemix
    :amari_scalemix
    :dkw_locmix
    :kde_locmix
    :amari_locmix
    :logspline
]

method_labels = (
    kde_scalemix = L"\textrm{Gauss-F-Loc } (\mathcal{S}\mathcal{N})",
    kde_locmix = L"\textrm{Gauss-F-Loc } (\mathcal{L}\mathcal{N})",
    dkw_scalemix = L"\textrm{DKW-F-Loc } (\mathcal{S}\mathcal{N})",
    dkw_locmix = L"\textrm{DKW-F-Loc } (\mathcal{L}\mathcal{N})",
    amari_scalemix = L"\textrm{AMARI } (\mathcal{S}\mathcal{N})",
    amari_locmix = L"\textrm{AMARI } (\mathcal{L}\mathcal{N})",
    logspline = L"\textrm{Logspline}",
)



function summary_df(ci_list, prior; methods_list = methods_list, method_labels = method_labels)
    _df = DataFrame(
        target = Any[],
        t = Float64[],
        method = Any[],
        method_label = Any[],
        cover = Bool[],
        simultaneous_cover = Bool[],
        ground_truth = Float64[],
        lower = Float64[],
        upper = Float64[],
        ci_length = Float64[],
        id = Int64[],
    )

    for (i, ci) in enumerate(ci_list)
        for method in methods_list
            ci_method = ci[method]
            if !isa(ci_method, Exception) && !any(isnan.(getproperty.(ci_method, :lower)))
                lower = getproperty.(ci_method, :lower)
                upper = getproperty.(ci_method, :upper)
                _targets = getproperty.(ci_method, :target)
                _ts = response.(location.(_targets))
                ground_truth = _targets.(prior)
                cover = lower .- 1e-6 .<= ground_truth .<= upper .+ 1e-6
                simultaneous_cover = all(cover)
                ci_length = upper .- lower
                append!(
                    _df,
                    DataFrame(
                        cover = cover,
                        simultaneous_cover = simultaneous_cover,
                        lower = lower,
                        upper = upper,
                        ci_length = ci_length,
                        ground_truth = ground_truth,
                        t = _ts,
                        target = _targets,
                        id = i,
                        method = method,
                        method_label = method_labels[method],
                    ),
                )
            end
        end
    end

    gdf = groupby(_df, [:target; :t; :method; :method_label])
    combined_gdf = combine(gdf, valuecols(gdf) .=> mean,
        valuecols(gdf) .=> z->length(unique(z)),
        nrow)
    combined_gdf = transform(
        combined_gdf,
        [:lower_mean, :upper_mean] =>
            ByRow((l, u) -> Empirikos.LowerUpperConfidenceInterval(lower = l, upper = u)) => :ci,
    )
    combined_gdf[!, :method_label] =  categorical(combined_gdf.method_label)
    levels!(combined_gdf.method_label, collect(values(method_labels)))
    combined_gdf
end

function simulation_plots(_df; _ylabel = "placeholder", scalemix = false)

    # Gauss-F-Loc, DKW-F-Loc, AMARI, logspline
    method_colors = [:darkorange :black :blue :darkred]
    method_alphas = [0.7 0.9 0.4 0.8]
    method_linestyle = [:solid :dash :solid  :dot]

    methods_locmix = [:logspline, :dkw_locmix, :kde_locmix, :amari_locmix]
    methods_scalemix = [:dkw_scalemix, :kde_scalemix, :amari_scalemix]

    if !scalemix
        _plot1 = @df filter(:method => ==(:kde_locmix), _df) plot(
            :t,
            :ci,
            fillcolor = :darkorange,
            fillalpha = 0.5,
            xlabel = L"z",
            ylabel = _ylabel,
            label = method_labels.kde_locmix,
        )
        @df filter(:method => ==(:dkw_locmix), _df) plot!(
            _plot1,
            :t,
            :ci,
            label = method_labels.dkw_locmix,
            show_ribbon = false,
            alpha = 0.9,
            color = :black,
        )
        @df filter(:method => ==(:amari_locmix), _df) plot!(
            _plot1,
            :t,
            :ci,
            label = method_labels.amari_locmix,
            show_ribbon = true,
            fillcolor = :blue,
            fillalpha = 0.4,
        )
        @df filter(:method => ==(:logspline), _df) plot!(
            _plot1,
            :t,
            :ci,
            label = method_labels.logspline,
            show_ribbon = false,
            linealpha = 1.0,
            linecolor = :darkred,
            linestyle = :dot,
        )
        @df filter(:method => ==(:amari_locmix), _df) plot!(
            _plot1,
            :t,
            :ground_truth_mean,
            label = "Ground truth",
            color = :black,
        )

        _plot3 = @df filter(:method => in(methods_locmix), _df) plot(
            :t,
            :cover_mean,
            group = :method_label,
            ylim = (0.0, 1.05),
            legend = :bottom,
            xlabel = L"z",
            ylabel = "Coverage",
            linecolor = method_colors,
            linealpha = method_alphas,
            linestyle = method_linestyle,
        )
        hline!(_plot3, [0.95], linestyle = :dot, label = nothing, color = :lightgrey)

        return _plot1, _plot3
    else

        _plot2 = @df filter(:method => ==(:kde_scalemix), _df) plot(
            :t,
            :ci,
            fillcolor = :darkorange,
            fillalpha = 0.5,
            xlabel = L"z",
            ylabel = _ylabel,
            label = method_labels.kde_scalemix,
        )
        @df filter(:method => ==(:dkw_scalemix), _df) plot!(
            _plot2,
            :t,
            :ci,
            label = method_labels.dkw_scalemix,
            show_ribbon = false,
            alpha = 0.9,
            color = :black,
        )
        @df filter(:method => ==(:amari_scalemix), _df) plot!(
            _plot2,
            :t,
            :ci,
            label = method_labels.amari_scalemix,
            show_ribbon = true,
            fillcolor = :blue,
            fillalpha = 0.4,
        )
        @df filter(:method => ==(:amari_scalemix), _df) plot!(
            _plot2,
            :t,
            :ground_truth_mean,
            label = "Ground truth",
            color = :black,
        )

        _plot4 = @df filter(:method => in(methods_scalemix), _df) plot(
            :t,
            :cover_mean,
            group = :method_label,
            ylim = (0.0, 1.05),
            legend = :bottom,
            xlabel = L"z",
            ylabel = "Coverage",
            linecolor = method_colors,
            linealpha = method_alphas,
            linestyle = method_linestyle,
        )
        hline!(_plot4, [0.95], linestyle = :dot, label = nothing, color = :lightgrey)

        return _plot2, _plot4
    end
end

spiky_postmean_df = summary_df(spiky_postmean, spiky_prior)
negspiky_postmean_df = summary_df(negspiky_postmean, negspiky_prior)
spiky_lfsr_df = summary_df(spiky_lfsr, spiky_prior)
negspiky_lfsr_df = summary_df(negspiky_lfsr, negspiky_prior)

# sanity checks
@test nrow(spiky_postmean_df) == 31*length(methods_list)
@test nrow(spiky_lfsr_df) == 31*length(methods_list)
@test nrow(negspiky_postmean_df) == 31*(length(methods_list)-3)
@test nrow(negspiky_lfsr_df) == 31*(length(methods_list)-3)
#

# Figure 4

spiky_postmean_locmix =
    simulation_plots(spiky_postmean_df, _ylabel = L"\EE{\mu \mid Z=z}", scalemix = false)
spiky_postmean_locmix[1]
spiky_postmean_locmix[2]

savefig(spiky_postmean_locmix[1], "spiky_postmean_locmix_intervals.tikz") #a)
savefig(spiky_postmean_locmix[2], "spiky_postmean_locmix_coverage.tikz")  #b)

negspiky_postmean_locmix =
    simulation_plots(negspiky_postmean_df, _ylabel = L"\EE{\mu \mid Z=z}", scalemix = false)

negspiky_postmean_locmix[1]
negspiky_postmean_locmix[2]

savefig(negspiky_postmean_locmix[1], "negspiky_postmean_locmix_intervals.tikz") #c)
savefig(negspiky_postmean_locmix[2], "negspiky_postmean_locmix_coverage.tikz") #d)

# Figure 5

spiky_lfsr_locmix =
    simulation_plots(spiky_lfsr_df, _ylabel = L"\PP{\mu \geq 0 \mid Z=z}", scalemix = false)
spiky_lfsr_locmix[1]
spiky_lfsr_locmix[2]

savefig(spiky_lfsr_locmix[1], "spiky_lfsr_locmix_intervals.tikz") #a)
savefig(spiky_lfsr_locmix[2], "spiky_lfsr_locmix_coverage.tikz")  # b)


negspiky_lfsr_locmix = simulation_plots(
    negspiky_lfsr_df,
    _ylabel = L"\PP{\mu \geq 0 \mid Z=z}",
    scalemix = false,
)

negspiky_lfsr_locmix[1]
negspiky_lfsr_locmix[2]

savefig(negspiky_lfsr_locmix[1], "negspiky_lfsr_locmix_intervals.tikz") #c)
savefig(negspiky_lfsr_locmix[2], "negspiky_lfsr_locmix_coverage.tikz")  #d)



# Figure 6



spiky_postmean_scalemix =
    simulation_plots(spiky_postmean_df, _ylabel = L"\EE{\mu \mid Z=z}", scalemix = true)
spiky_postmean_scalemix[1]
spiky_postmean_scalemix[2]

savefig(spiky_postmean_scalemix[1], "spiky_postmean_scalemix_intervals.tikz") #a)
savefig(spiky_postmean_scalemix[2], "spiky_postmean_scalemix_coverage.tikz")  #b)



spiky_lfsr_scalemix =
    simulation_plots(spiky_lfsr_df, _ylabel = L"\PP{\mu \geq 0 \mid Z=z}", scalemix = true)
spiky_lfsr_scalemix[1]
spiky_lfsr_scalemix[2]

savefig(spiky_lfsr_scalemix[1], "spiky_lfsr_scalemix_intervals.tikz") # c)
savefig(spiky_lfsr_scalemix[2], "spiky_lfsr_scalemix_coverage.tikz")  # d)




varying_dof = load("simulation_expfamily.jld2", "ci_list")

methods_exp = collect(2:13)

method_labels_exp = Dict(methods_exp .=> string.(methods_exp))

varying_dof_df =
    summary_df(
        varying_dof,
        negspiky_prior;
        methods_list = methods_exp,
        method_labels = method_labels_exp,
    ) |> u -> filter(:method => <(13), u)

# sanity check

sim_ebci_5 = filter(:method => ==(:logspline), filter(:t => ==(2.0), negspiky_lfsr_df))
sim_expfam_5 = filter(:method => ==(5), varying_dof_df)
@test sim_expfam_5.cover_mean ≈ sim_ebci_5.cover_mean
@test sim_expfam_5.ci_length_mean ≈ sim_ebci_5.ci_length_mean


negspiky_dof_filter =
    filter(:method => !=(:logspline), filter(:t => ==(2.0), negspiky_lfsr_df))




varying_dof_plot = @df negspiky_dof_filter plot(
    :ci_length_mean,
    :cover_mean,
    group = :method_label,
    seriestype = :scatter,
    marker = [:utriangle :rtriangle :ltriangle],
    markersize = 10,
    color = [:darkorange :black :blue],
    alpha = 0.5,
    markerstrokealpha = 0.7,
    size = (550, 300),
    ylim = (-0.1, 1.1),
    xlim = (0.05, 0.6),
    markerstrokewidth = 1.3,
)

plot!(
    varying_dof_plot,
    varying_dof_df.ci_length_mean,
    varying_dof_df.cover_mean,
    seriestype = :scatter,
    markersize = 10,
    label = "Logspline",
    color = :darkblue,
    bg_legend = :transparent,
    fg_legend = :transparent,
    alpha = range(0.2, 0.5, length = 11),
    markerstrokealpha = 0.7,
    markerstrokecolor = :black,
    markerstrokewidth = 1.3,
    xlabel = "Expected CI length",
    ylabel = "Coverage",
)


annotate!(
    varying_dof_plot,
    varying_dof_df.ci_length_mean,
    varying_dof_df.cover_mean,
    text.(varying_dof_df.method_label, 13),
    alpha = 0.6,
    color = :black,
)

hline!(varying_dof_plot, [0.95], color = :lightgrey, linestyle = :dash, label = "")

savefig(varying_dof_plot, "varying_dof_plot.tikz")
