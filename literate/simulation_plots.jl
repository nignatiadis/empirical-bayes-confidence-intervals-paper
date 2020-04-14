using DrWatson
@quickactivate "MCEBPaper"

using Query
using TypedTables
using MinimaxCalibratedEBayes
using ExponentialFamilies
using MosekTools
using LaTeXStrings
using FileIO
using EBayes


const MCEB = MinimaxCalibratedEBayes

include(srcdir("plotting_helpers.jl"))

twintowers = load(datadir("sims","twintowers.jld2"), "res")
bimodal = load(datadir("sims","bimodal.jld2"), "res")
bimodal = filter( x-> x != "error", bimodal)

unimodal = load(datadir("sims","unimodal_postmean.jld2"), "res")
varying_dof = load(datadir("sims","varyingdof_lfsr.jld2"), "res")

res_tuples = vcat([twintowers; bimodal; unimodal; varying_dof]...)

res_tuples_process = @from i in res_tuples begin
       @select  {i..., realized_error = i.estimated_target - i.target_value,
                       covers = i.lower_ci <= i.target_value <= i.upper_ci,
											 ci_width = i.upper_ci - i.lower_ci,					 
					   prior_name = prior_string(i.true_dist),
					   method_name = method_string(i.method),
					   method_params = method_params(i.method)}
       @collect Table
end


res_df = @from i in res_tuples_process begin
         @group i by {i.target, i.prior_name, i.method_name, i.method_params, i.target_value} into g
         @select    {key(g)...,
	                estimated_target = mean(g.estimated_target),
	                coverage = mean(g.covers),
					realized_bias = mean(g.realized_error),
					realized_std = std(g.estimated_target),
					estimated_bias  = mean(g.estimated_bias),
					estimated_std  = mean(g.estimated_std),
					ci_width = mean(g.ci_width),
					lower_ci = mean(g.lower_ci),
					upper_ci = mean(g.upper_ci),
					true_dist = g.true_dist[1],
					target_string = MCEB.pretty_label(key(g).target),
					target_type = target_string(key(g).target),
					target_location = response(key(g).target)}
         @collect Table
end


bimodal_postmean = res_df |> @filter(_.target_type == "Posterior Mean" &&
                                     _.prior_name == "Bimodal") |> Table

twotower_postmean = res_df |> @filter(_.target_type == "Posterior Mean" &&
                                       _.prior_name == "Logspline Two Towers") |> Table


bimodal_lfsr = res_df |> @filter(_.target_type == "LFSR" &&
								 _.prior_name == "Bimodal") |> Table

twotower_lfsr = res_df |> @filter(_.target_type == "LFSR" &&
								   _.prior_name == "Logspline Two Towers") |> Table


unimodal_lfsr = res_df |> @filter(_.target_type == "LFSR" && 
                                  _.prior_name == "Unimodal" &&
								  (_.method_name == "MCEB" || _.method_params == 5))|> Table 
								  #
unimodal_postmean = res_df |> @filter(_.target_type == "Posterior Mean" &&
								     _.prior_name == "Unimodal") |> Table

      

def_size = (850,260)



pl_bimodal_lfsr = simulationplot(bimodal_lfsr; size=def_size, density_ylim =(0,1.2),
                                 bands_legend=:topleft);
savefig(pl_bimodal_lfsr, "pl_bimodal_lfsr.pdf")

pl_unimodal_lfsr = simulationplot(unimodal_lfsr; size=def_size, 
                                  bands_legend=:topleft, density_ylim =(0,1.7),
								  titles = ["d)"; "e)"; "f)"])
savefig(pl_unimodal_lfsr, "pl_unimodal_lfsr.pdf")


pl_twotower_lfsr = simulationplot(twotower_lfsr; size=def_size, density_xlim = (-4,6),
                                   density_ylim = (0,0.25),
                                   bands_legend=:topleft, density_legend=:topright);

savefig(pl_twotower_lfsr, "pl_twotower_lfsr.pdf")



pl_twotower_postmean = simulationplot(twotower_postmean; size=def_size, density_xlim = (-4,6),
                                   density_ylim = (0,0.25),
                                   bands_legend=:topleft, density_legend=:topright,
								   titles = ["g)"; "h)"; "i)"]);

savefig(pl_twotower_postmean, "pl_twotower_postmean.pdf")


pl_bimodal_postmean = simulationplot(bimodal_postmean; size=def_size, density_ylim =(0,1.2),
                                 bands_legend=:topleft);
savefig(pl_bimodal_postmean, "pl_bimodal_postmean.pdf")

pl_unimodal_postmean = simulationplot(unimodal_postmean; size=def_size, 
                                  bands_legend=:topleft, density_ylim =(0,1.7),
								  titles = ["d)"; "e)"; "f)"])
savefig(pl_unimodal_postmean, "pl_unimodal_postmean.pdf")




# # Figure 2

gmodel_lfsr = res_df |> @filter(_.target_type == "LFSR" && 
                                _.prior_name == "Unimodal") |> Table

gmodel_lfsr_collapse = @from i in gmodel_lfsr begin
	                   @group i by {i.target_type, i.prior_name, i.method_name, i.method_params} into g
	                   @select    {key(g)...,
			                       coverage = mean(g.coverage),
			                       ci_width = mean(g.ci_width)}
					   @collect Table end 


gmodel_lfsr_collapse_gmodel = gmodel_lfsr_collapse |> @filter(_.method_name == "G-model")|> Table
gmodel_lfsr_collapse_mceb = gmodel_lfsr_collapse |> @filter(_.method_name == "MCEB")|> Table
plot(gmodel_lfsr_collapse_gmodel.method_params, gmodel_lfsr_collapse_gmodel.coverage, seriestype=:scatter)


pgfplotsx()
varying_dof_plot = plot(gmodel_lfsr_collapse_mceb.ci_width, gmodel_lfsr_collapse_mceb.coverage,
                        seriestype=:scatter, 
                        marker=:utriangle, markersize=20, label="MCEB", color="#CD5496", markerstrokewidth=1.3)
varying_dof_plot = plot!(varying_dof_plot, 
                        gmodel_lfsr_collapse_gmodel.ci_width, gmodel_lfsr_collapse_gmodel.coverage,
                        seriestype=:scatter,
			            markersize=20, label="G-model", legend=:bottomright, color=:darkblue,
			            bg_legend=:transparent, fg_legend=:transparent, alpha = range(0.2, 0.5, length=14),
						markerstrokealpha = 1, markerstrokecolor =:black,
			            ylim=(-0.06,1.1), size=(550,300), markerstrokewidth=1.3,
						xlabel="Average CI width", ylabel="Coverage")
annotate!(gmodel_lfsr_collapse_gmodel.ci_width, 
          gmodel_lfsr_collapse_gmodel.coverage, 
		  string.(gmodel_lfsr_collapse_gmodel.method_params), color=:black)

hline!(varying_dof_plot, [0.9],  color = :grey, linestyle = :dash, label="")

savefig("gmodel_varying_dof.svg")
savefig("gmodel_varying_dof2.pdf")




#' Finally plot worst case bias etc.


bimodal_postmean_mceb = bimodal_postmean |> @filter(_.method_name == "MCEB") |> Table
bimodal_lfsr_mceb = bimodal_lfsr |> @filter(_.method_name == "MCEB") |> Table
twotower_postmean_mceb = twotower_postmean |> @filter(_.method_name == "MCEB") |> Table
twotower_lfsr_mceb = twotower_lfsr |> @filter(_.method_name == "MCEB") |> Table
unimodal_lfsr_mceb = unimodal_lfsr |> @filter(_.method_name == "MCEB") |> Table
unimodal_postmean_mceb = unimodal_postmean |> @filter(_.method_name == "MCEB") |> Table

 


function plot_bias_std(data, legend=:topleft)
	dbn_name = data.prior_name[1]
	@show target_name = MCEB.pretty_label(data.target[1])[5:end]
	tmp_title = dbn_name*","*target_name
	xs = data.target_location
	std_error = data.realized_std
	bias = abs.(data.realized_bias)
	max_bias = abs.(data.estimated_bias)
	rmse = sqrt.(bias.^2 + std_error.^2)
	max_rmse = sqrt.(max_bias.^2 + std_error.^2)
	ci_width = data.ci_width
	tmp_plot = plot(xs, rmse, fg_legend=:transparent,
	                bg_legend=:transparent, legend=legend,
					linewidth=2, label="Std. error", linestyle=:solid, 
	                color=RGB(68/255, 69/255, 145/255), title=tmp_title)
	plot!(tmp_plot, xs, rmse, linewidth=2, label="|Bias|", linestyle=:dash, color= RGB(205/255, 84/255, 150/255))
	plot!(tmp_plot, xs, ci_width,linewidth=2, label="Max. Bias", linestyle=:dot, color=RGB(132/255, 193/255, 216/255))
	tmp_plot
end

function plot_bias_std(data, legend, )
	dbn_name = data.prior_name[1]
	target_name = MCEB.pretty_label(data.target[1])
	tmp_title = dbn_name
	xs = data.target_location
	std_error = data.realized_std
	bias = abs.(data.realized_bias)
	max_bias = abs.(data.estimated_bias)
	rmse = sqrt.(bias.^2 + std_error.^2)
	max_rmse = sqrt.(max_bias.^2 + std_error.^2)
	ci_width = data.ci_width./2
	tmp_plot = plot(xs, rmse, ylim=(0,0.4), xlabel=L"x", ylabel=target_name, fg_legend=:transparent, 
	                bg_legend=:transparent, legend=legend,
					linewidth=2, label="RMSE", linestyle=:solid, 
	                color=RGB(68/255, 69/255, 145/255), title=tmp_title)
	plot!(tmp_plot, xs, max_rmse, linewidth=2, label="Max. RMSE", linestyle=:dash, color= RGB(205/255, 84/255, 150/255))
	plot!(tmp_plot, xs, ci_width,linewidth=2, label="Half CI width", linestyle=:dot, color=RGB(132/255, 193/255, 216/255))
	tmp_plot
end

legends = [:topleft, nothing, nothing, nothing, nothing, nothing]

all_plots = plot_bias_std.([bimodal_postmean_mceb,
                  unimodal_postmean_mceb,
				  twotower_postmean_mceb,
				  bimodal_lfsr_mceb, 
				  unimodal_lfsr_mceb,
				  twotower_lfsr_mceb],  legends)
		

def_size = (900,440)
		
				  
plot(all_plots..., layout=(2,3),size=def_size)				  
savefig("rmse_ciwidth_plots.pdf")			  
plot_bias_std(unimodal_lfsr_mceb)
 
plot(xs, bimodal_lfsr_mceb.realized_std, label="std")
plot(xs, abs.(bimodal_lfsr_mceb.realized_bias), label="std")

plot(xs, bimodal_postmean_mceb.realized_std, label="std")
plot(xs, abs.(bimodal_postmean_mceb.realized_bias), label="std")




twotower_lfsr 


								  #
unimodal_postmean 
      