using StatsPlots
using RecipesBase
using Colors




function prior_string(dist::MixtureModel)
	_comps = components(dist)
	_probs = probs(dist)
	iwbimod = EBayes.IWBimod().distribution
	iwunimod = EBayes.IWUnimod().distribution
	if _probs == probs(iwbimod) && _comps == components(iwbimod)
		return("Bimodal")
	elseif _probs == probs(iwunimod) && _comps == components(iwunimod)
		return("Unimodal")
	else
		return("Mixture")
	end
end

prior_string(::ContinuousExponentialFamily) = "Smoothed Twin-Towers"
prior_string(dbn) = string(typeof(dbn))[1:10]

method_string(::ExponentialFamilyDeconvolutionMLE) = "G-model"
method_string(::MinimaxCalibratorOptions) = "MCEB"

method_params(cefm_mle::ExponentialFamilyDeconvolutionMLE) = cefm_mle.cefm.Q_dim
method_params(cefm_mle) = 0

target_string(::MCEB.PosteriorTarget{<:PosteriorMeanNumerator}) = "Posterior Mean"
target_string(::MCEB.PosteriorTarget{<:LFSRNumerator}) = "LFSR"






@userplot SimulationPlot
# "#550133"

# mceb color "#EBA415"
# gmodel color "#550133"
@recipe function f(h::SimulationPlot; method_names= ["MCEB";"G-model"],
	                                  density_xlim = nothing,
									  titles = ["a)"; "b)"; "c)"],
									  gmodel_alpha = 0.5,
									  gmodel_color =  "#C68642",# "#C2C2B4", #"#550133", 
									  mceb_color = "#F6CD7F") #"#EBA415")#)
    res_df = first(h.args)
	mceb_df = res_df |> @filter(_.method_name == method_names[1]) |> Table
	gmodel_df = res_df |> @filter(_.method_name == method_names[2]) |> Table
    
	true_prior = res_df.true_dist[1]
	prior_name = res_df.prior_name[1]

	layout := @layout [panel1 panel2 panel3]

	fg_legend --> :transparent
	bg_legend --> :transparent
	xlim --> extrema(res_df.target_location) .+ (-0.043,0.01)
	
    # Plot density
    @series begin
		title := titles[1]
        seriestype := :path
        subplot := 1
		if !isnothing(density_xlim)
			xlim := density_xlim
		end 
        #xlim := extrema(x_grid)
		label := prior_name
		legend := :topright
		color --> "#018AC4"
		xguide := L"\mu"
		yguide := L"Density $g(\mu)$"
		if isa(true_prior, MixtureModel)
			components := false
		end
        true_prior
    end

	@series begin
	 	subplot := 2
		fillrange := mceb_df.upper_ci
		color := mceb_color
		label := method_names[1]*" Band"
		mceb_df.target_location, mceb_df.lower_ci
	end
	
	@series begin
		subplot := 2
		fillrange := gmodel_df.upper_ci
		color := gmodel_color
		label := method_names[2]*" Band"
		gmodel_df.target_location, gmodel_df.lower_ci
	end

	
	@series begin
		seriestype := :path
		title := titles[2]
		subplot := 2
		xguide := L"x"
		label := "true target"
		ylabel --> MCEB.pretty_label(res_df.target[1])
		linestyle := :dot
		legend --> :bottomright
		color := :black
		mceb_df.target_location, mceb_df.target_value
	end
	# Coverage 
	@series begin
		subplot :=3
		seriestype := :path
		label --> method_names[1]
		color --> mceb_color     #"#EBA415"#"#EBC915"
		alpha --> 1
		linewidth --> 1.5
		mceb_df.target_location, mceb_df.coverage
	end 
	@series begin
		subplot :=3 
		seriestype := :hline
		label := nothing
		color := :grey
		linestyle := :dash
		[0.9]
	end
	@series begin
		subplot :=3
		title := titles[3]
		seriestype := :path
		xguide := L"x"
		ylim --> (0,1.3)
		color --> gmodel_color#"#550133"
		#alpha -> 0.4
		linewidth --> 1.5
		yguide --> "Coverage"
		legend --> :topright
		label --> method_names[2]
		gmodel_df.target_location, gmodel_df.coverage
	end 	
end