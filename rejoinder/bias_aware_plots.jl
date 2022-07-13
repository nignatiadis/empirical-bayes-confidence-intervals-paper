using Pkg
Pkg.activate(".")
using Distributions
using Plots
using LaTeXStrings
using Empirikos

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


bs = 0:0.005:2
α = 0.05
physicist_halfci = quantile(Normal(), 1-α/2) * sqrt.(1 .+ abs2.(bs))
add_bias_onesided = quantile(Normal(), 1-α) .+ bs
im = [Empirikos.gaussian_ci(1; maxbias = b, α=α) for b in bs]

# Panel a) of Figure 1
interval_length_plot = plot(
    bs,
    [im physicist_halfci add_bias_onesided],
    color = [:darkblue :teal :purple],
    label = [L"t_{\alpha}(B,1)" L"t_{\alpha}^{\mathrm{phys}}(B,1)" L"t_{\alpha}^{\mathrm{one}}(B,1)"],
    linestyle = [:solid :dash :dot],
    legend = :bottomright,
    xlabel = L"B",
    ylabel = L"|t|",
    ylim = (0, 4.2),
    xlim = (0, 2),
    yticks = [1.64; 1.96; 3; 4],
    thickness_scaling = 1.6,
    size = (380, 280)
)

savefig(interval_length_plot, "interval_length_plot.tikz")



bs_bias = 0:0.01:50
αs = [0.05; 0.05/10; 0.05/100; 0.05/1000]

im_varying_α = [Empirikos.gaussian_ci(1; maxbias = b, α=α) for b in bs_bias, α in αs]
im_varying_α_normalized = im_varying_α[:,2:4] ./ im_varying_α[:,1]


multiplicity_plot = plot(
    bs_bias,
    im_varying_α_normalized,
    color = [:darkblue :teal :purple],
    label = [L"K=10" L"K=100" L"K=1000"],
    linestyle = [:solid :dash :dot],
    legend = :topright,
    xlabel = L"B",
    ylabel = L"t_{\alpha/K}(B,1)/t_{\alpha}(B,1)",
    xlim = (0,50),
    thickness_scaling = 1.6,
    size = (380, 280)
)

savefig(multiplicity_plot, "multiplicity_plot.tikz")
