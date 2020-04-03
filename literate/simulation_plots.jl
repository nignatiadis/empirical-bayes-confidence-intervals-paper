using DrWatson
@quickactivate "MCEBPaper"

using Query
using TypedTables
using MinimaxCalibratedEBayes
using ExponentialFamilies
using MosekTools
using LaTeXStrings
using FileIO



const MCEB = MinimaxCalibratedEBayes

include(srcdir("plotting_helpers.jl"))

res = load(datadir("sims","twintowers.jld2"), "res")
res_tuples = vcat(res...)

method_string(::MinimaxCalibratedEBayes)
	
method_string(::ExponentialFamilyDeconvolutionMLE) = "G-model"
method_string(::MinimaxCalibratorOptions) = "MCEB"

method_params(cefm_mle::ExponentialFamilyDeconvolutionMLE) = cefm_mle.cefm.Q_dim
method_params(cefm_mle) = NaN

res_tuples_process = @from i in res_tuples begin
       @select  {i..., realized_error = i.estimated_target - i.target_value,
                       covers = i.lower_ci <= i.target_value <= i.upper_ci,
											 ci_width = i.upper_ci - i.lower_ci,					 
                       location = response(i.target),
					   prior_name = string(typeof(i.true_dist))[1:10],
					   method_name = method_string(i.method),
					   method_params = method_params(i.method)}
       @collect Table
end



res_df = @from i in res_tuples_process begin
       @group i by {i.target, i.prior_name, i.method_name, i.target_value} into g
       @select {key(g)...,
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
								target_location = response(key(g).target)}
       @collect Table
end














gr()
pl1 = simulationplot(mydf_lfsr; density_xlim = (-4,6),
                      size=(800,300))
					  
savefig(pl1, "pl1.svg")					  
savefig(pl1, "pl1.pdf")					  

