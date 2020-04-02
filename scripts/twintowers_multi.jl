using DrWatson
@quickactivate "MCEBPaper"

idx = 1#ARGS[1]
nreps_per_proc = 50


include(srcdir("mceb_sim_eval.jl"))
	
using MosekTools

#--- Global options----------------------------------------------------------
alpha_level = 0.9

#--- Targets------------------------------------------------------------------
target_grid = -3:0.2:3
ebayes_targets = [LFSR.(StandardNormalSample.(target_grid));
	                  PosteriorMean.(StandardNormalSample.(target_grid))]


#--- Data generation settings-------------------------------------------------
expfamily = ContinuousExponentialFamilyModel(Uniform(-4, 6), collect(range(-4,6, step=0.01));
	                                             df=8, scale=true)
											 
n = 10_000
α_smoothed_twin_tower =  [10; 0.0; 6.0; 9.0; -3.0; -2.0; -8.0; -1.0]
smoothed_twin_tower = expfamily(α_smoothed_twin_tower)


#-- Methods compared ----------------------------------------------------------
##---- 1) logspline MLE method
expfamily_solver = MCEB.ExponentialFamilyDeconvolutionMLE(cefm = expfamily, c0=0.0001,
														 marginal_grid = -6:0.01:8)	

##-----2) MCEB method							
gcal = GaussianMixturePriorClass(0.2, -6:0.025:6, Mosek.Optimizer)
																												 
mceb_options = MinimaxCalibratorOptions(prior_class = gcal, 
										marginal_grid =  -6:0.05:6,
										pilot_options =  MCEB.ButuceaComteOptions(),
										cache_target = true)																												 																										 

##------ Put mehtods into array										
ebayes_methods = [expfamily_solver;
                  mceb_options]
	
##------ add everything into closure	
custom_rng = (1 + nreps_per_proc*(idx-1)):(nreps_per_proc*idx)

for i in custom_rng
	@show i
	res = mceb_sim_eval(i; eb_prior = smoothed_twin_tower,
				  	 n = n,
				  	 alpha_level = alpha_level,
				  	 eb_methods = ebayes_methods,
				  	 targets = ebayes_targets)
	safesave(datadir("sims","twintowers","tmp_twintowers_$i.jld2"), Dict("res"=>res))			 
end 
				  				  

