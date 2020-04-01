using ApproxFun
using ExponentialFamilies
using MinimaxCalibratedEBayes
const MCEB = MinimaxCalibratedEBayes

function mceb_sim_eval(i; eb_prior = eb_prior,
	                n = 10_000,
                  alpha_level = 0.9,
                  eb_methods = eb_methods,
                  targets = targets)
   res_tuples = []
  
   μs = rand(eb_prior, n) #
   Zs = StandardNormalSample.(μs .+ randn(n))
  
    for method in eb_methods
        method_setup = fit(method, Zs)
        for t in targets
            true_target = t(eb_prior)
            method_fit = fit(method_setup, t) 
            result_nt = MCEB.target_bias_std(t, method_fit)
            ci_nt = confint(t, method_fit; level=alpha_level)
       
            res_nt = (target = t, 
                      true_dist = eb_prior,
					  n = n,
                      iteration = i,
                      method = method,
                      target_value = true_target, 
                      result_nt..., 
                      lower_ci = ci_nt[1],
                      upper_ci = ci_nt[2],
					  host = gethostname(),
					  procid = myid())
                
            push!(res_tuples, res_nt)
        end
    end
	res_tuples
end