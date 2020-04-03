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

twintower_postmean = res_df |> @filter(_.target_type == "Posterior Mean" &&
                                       _.prior_name == "Smoothed Twin-Towers") |> Table


bimodal_lfsr = res_df |> @filter(_.target_type == "LFSR" &&
								 _.prior_name == "Bimodal") |> Table

twintower_lfsr = res_df |> @filter(_.target_type == "LFSR" &&
								   _.prior_name == "Smoothed Twin-Towers") |> Table


unimodal_lfsr = res_df |> @filter(_.target_type == "LFSR" && 
                                  _.prior_name == "Unimodal" &&
								  (_.method_name == "MCEB" || _.method_params == 5))|> Table 
								  #
unimodal_postmean = res_df |> @filter(_.target_type == "Posterior Mean" &&
								     _.prior_name == "Unimodal") |> Table

      

def_size = (800,240)

pl1 = simulationplot(unimodal_lfsr; size=def_size)

pl_unimodal_postmean = simulationplot(unimodal_postmean; size=def_size)

savefig("pl_unimodal_postmean.svg")
savefig("pl_unimodal_postmean.pdf")



pl1 = simulationplot(bimodal_postmean; size=def_size)

pl1 = simulationplot(bimodal_lfsr; size=def_size)

pl1 = simulationplot(twintower_postmean; density_xlim = (-4,6), size=def_size)

pl1 = simulationplot(twintower_lfsr; density_xlim = (-4,6), def_size)
					  
savefig(pl1, "pl1.svg")					  
savefig(pl1, "pl1.pdf")					  



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

