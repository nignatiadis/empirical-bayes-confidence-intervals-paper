using DrWatson
@quickactivate "MCEBPaper"

using Distributed
using ClusterManagers

addprocs(6) 


scriptloc = srcdir("mceb_sim_eval.jl")

#--- get everything on remote nodes
@everywhere using DrWatson
@everywhere @quickactivate "MCEBPaper"
@eval @everywhere include($scriptloc)
@everywhere begin
	
	using MosekTools
	using EBayes
	
	#--- Global options----------------------------------------------------------
	nreps = 500
 	alpha_level = 0.9

	#--- Targets------------------------------------------------------------------
	target_grid = -3:0.25:3
	ebayes_targets = LFSR.(StandardNormalSample.(target_grid))

	#--- Data generation settings-------------------------------------------------
											
	n = 10_000
	true_dist = EBayes.IWUnimod().distribution



	#-- Methods compared ----------------------------------------------------------
	ebayes_methods = []
	##---- 1) logspline MLE methods with different splines
	
	
	logspline_degs = 2:15#:15
	
	for deg in logspline_degs
		cefm = ContinuousExponentialFamilyModel(Uniform(-3.6,3.6), collect(-3.6:0.01:3.6); df = deg)
		expfamily_solver = MCEB.ExponentialFamilyDeconvolutionMLE(cefm = cefm, c0=0.001,
															      marginal_grid = -6:0.02:6)
		push!(ebayes_methods, expfamily_solver)														  
	end
	##-----2) MCEB method							
	gcal = GaussianMixturePriorClass(0.2, -3:0.025:3, Mosek.Optimizer, (QUIET=true,))
																													 
	mceb_options = MinimaxCalibratorOptions(prior_class = gcal, 
											marginal_grid =  -6:0.05:6,
											pilot_options =  MCEB.ButuceaComteOptions(),
											cache_target = true)																												 																										 

	push!(ebayes_methods, mceb_options)
		
	##------ add everything into closure	
	function mceb_sim_eval_closure(i; save_intermediate=true)
		println("i=$i")
		println("myid=$(myid())")
		res = mceb_sim_eval(i; eb_prior = true_dist,
					  	         n = n,
					  	         alpha_level = alpha_level,
					  	         eb_methods = ebayes_methods,
					  	         targets = ebayes_targets)
		if save_intermediate						 
			safesave(datadir("sims","varyingdof_lfsr","varyingdof_lfsr_$i.jld2"), Dict("res"=>res))				  				  
		end				 
		res
	end 
end 
				  
# start simulations				  
res = pmap(mceb_sim_eval_closure,
			  Base.OneTo(nreps);
			  on_error = x->"error")		

			  
			  
safesave(datadir("sims","varyingdof_lfsr.jld2"), Dict("res"=>res))

#@eval @load $(datadir("sims","quick_check.jld2"))

