using JLD2
using Empirikos
using DataFrames
using LaTeXStrings
using StatsPlots
using SpecialExponentialFamilies
using FileIO

begin
    pgfplotsx()
    deleteat!(PGFPlotsX.CUSTOM_PREAMBLE, Base.OneTo(length(PGFPlotsX.CUSTOM_PREAMBLE)))
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

# ╔═╡ 17be2c0c-4bf5-11eb-25ca-cfe4e5fb5485
theme(
    :default;
    background_color_legend = :transparent,
    foreground_color_legend = :transparent,
    grid = nothing,
    frame = :box,
    thickness_scaling = 1.3,
)


spiky_prior = Empirikos.AshPriors[:Spiky]
negspiky_prior = MixtureModel([Normal(-0.25, 0.25), Normal(0, 1.0)], [0.8, 0.2])

tts = -3:0.01:3
plot(tts, pdf.(prior3, tts))
plot(tts, pdf.(prior3, StandardNormalSample.(tts)))
plot!(tts, pdf.(prior, StandardNormalSample.(tts)))


negspiky_lfsr = load("negspiky_lfsr.jld2", "ci_list")
spiky_lfsr = load("spiky_lfsr.jld2", "ci_list")
negspiky_postmean = load("negspiky_postmean.jld2", "ci_list")
spiky_postmean = load("spiky_postmean.jld2", "ci_list")


methods = [
    :dkw_scalemix
    :kde_scalemix
    :lam_scalemix
    :dkw_locmix
    :kde_locmix
    :lam_locmix
    :logspline
]

method_labels = (
    dkw_scalemix = L"\textrm{DKW-F-Loc } (\mathcal{G}=\mathcal{S}\mathcal{N})",
    dkw_locmix = L"\textrm{DKW-F-Loc } (\mathcal{G}=\mathcal{L}\mathcal{N})",
    kde_scalemix = L"\textrm{KDE-F-Loc } (\mathcal{G}=\mathcal{S}\mathcal{N})",
    kde_locmix = L"\textrm{KDE-F-Loc } (\mathcal{G}=\mathcal{L}\mathcal{N})",
    lam_scalemix = L"\textrm{Amari } (\mathcal{G}=\mathcal{S}\mathcal{N})",
    lam_locmix = L"\textrm{Amari } (\mathcal{G}=\mathcal{L}\mathcal{N})",
    logspline = L"\textrm{Logspline}",
)


function summary_df(ci_list, prior; methods = methods, method_labels = method_labels)
    _df = DataFrame(
        target = Any[],
        t = Float64[],
        method = Symbol[],
        method_label = AbstractString[],
        cover = Bool[],
        ground_truth = Float64[],
        lower = Float64[],
        upper = Float64[],
        ci_length = Float64[],
        id = Int64[],
    )

    for (i, ci) in enumerate(ci_list)
        for method in methods
            ci_method = ci[method]
            if !isa(ci_method, Exception)
                lower = getproperty.(ci_method, :lower)
                upper = getproperty.(ci_method, :upper)
                _targets = getproperty.(ci_method, :target)
                _ts = response.(location.(_targets))
                ground_truth = _targets.(prior)
                cover = lower .- 1e-6 .<= ground_truth .<= upper .+ 1e-6
                ci_length = upper .- lower
                append!(
                    _df,
                    DataFrame(
                        cover = cover,
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
    combined_gdf = combine(gdf, valuecols(gdf) .=> mean, nrow)
    combined_gdf = transform(
        combined_gdf,
        [:lower_mean, :upper_mean] =>
            ByRow((l, u) -> Empirikos.LowerUpperConfidenceInterval(lower = l, upper = u)) => :ci,
    )
    combined_gdf
end

function simulation_plots(_df, _ylabel = "bla", scalemix = false)

    method_colors = [:blue :darkorange :black :darkred]
    method_alphas = [0.4 0.5 0.9 0.8]
    method_linestyle = [:solid :solid :dash :dot]

    methods_locmix = [:logspline, :dkw_locmix, :kde_locmix, :lam_locmix]
    methods_scalemix = [:logspline, :dkw_scalemix, :kde_scalemix, :lam_scalemix]

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
        @df filter(:method => ==(:lam_locmix), _df) plot!(
            _plot1,
            :t,
            :ci,
            label = method_labels.lam_locmix,
            show_ribbon = true,
            fillcolor = :blue,
            fillalpha = 0.4,
        )
        @df filter(:method => ==(:logspline), _df) plot!(
            _plot1,
            :t,
            :ci,
            label = "Logspline",
            show_ribbon = false,
            linealpha = 1.0,
            linecolor = :darkred,
            linestyle = :dot,
        )
        @df filter(:method => ==(:lam_locmix), _df) plot!(
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
            legend = :bottomright,
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
        @df filter(:method => ==(:lam_scalemix), _df) plot!(
            _plot2,
            :t,
            :ci,
            label = method_labels.lam_scalemix,
            show_ribbon = true,
            fillcolor = :blue,
            fillalpha = 0.4,
        )
        @df filter(:method => ==(:lam_scalemix), _df) plot!(
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
            legend = :bottomright,
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



_df = spiky_postmean_df

bla = simulation_plots(_df)

bla[1]
bla[2]

negspiky_lfsr_df = summary_df(negspiky_lfsr)
spiky_lfsr_df = summary_df(spiky_lfsr, spiky_prior)





plot!(
    postmean_locmix_plot,
    [-3.0; 3.0],
    [-3.0; 3.0],
    seriestype = :line,
    linestyle = :dot,
    label = nothing,
    color = :lightgrey,
)
plot!(postmean_locmix_plot, xlabel = L"z", ylabel = L"\EE{\mu \mid Z=z}", size = (380, 280))



@df spiky_lfsr_df plot(:t, :cover_mean, group = :method, ylim = (0.0, 1.05))




@df spiky_postmean_df plot(
    :t,
    :ci_length_mean,
    group = :method,
    ylim = (0.0, 1.05),
    legend = :bottomright,
)
