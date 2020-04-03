using StatsPlots
using RecipesBase

@userplot SimulationPlot

@recipe function f(h::SimulationPlot; method_names= ["MCEB";"G-model"],
	                                  density_xlim = nothing,
									  titles = ["a)"; "b)"; "c)"])
    res_df = first(h.args)
	mceb_df = res_df |> @filter(_.method_name == method_names[1]) |> Table
	gmodel_df = res_df |> @filter(_.method_name == method_names[2]) |> Table
    
	true_prior = res_df.true_dist[1]
	layout := @layout [panel1 panel2 panel3]

	fg_legend --> :transparent
	bg_legend --> :transparent
	xlim --> extrema(res_df.target_location)
	
    # Plot density
    @series begin
		title := titles[1]
        seriestype := :path
        subplot := 1
		if !isnothing(density_xlim)
			xlim := density_xlim
		end 
        #xlim := extrema(x_grid)
		color --> "#018AC4"
		xguide := L"\mu"
		yguide := L"g(\mu)"
        true_prior
    end

	@series begin
		subplot := 2
		color := "#EBA415"
		fillalpha := 1.0
		ribbon --> (mceb_df.estimated_target .- mceb_df.lower_ci, mceb_df.upper_ci .- mceb_df.estimated_target)
		mceb_df.target_location, mceb_df.estimated_target
	end
	
	@series begin
		subplot := 2
		color := "#550133"
		alpha := 0.4
		ribbon --> (gmodel_df.estimated_target .- gmodel_df.lower_ci, gmodel_df.upper_ci .- gmodel_df.estimated_target)
		gmodel_df.target_location, gmodel_df.estimated_target
	end
	
	@series begin
		seriestype := :path
		title := titles[2]
		subplot := 2
		xguide := L"x"
		label := "true target"
		ylabel --> MCEB.pretty_label(res_df.target[1])
		linestyle := :dot
		color := :black
		mceb_df.target_location, mceb_df.target_value
	end
	# Coverage 
	@series begin
		subplot :=3
		seriestype := :path
		label --> method_names[1]
		color --> "#EBA415"#"#EBC915"
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
		color --> "#550133"
		alpha -> 0.4
		linewidth --> 1.5
		yguide --> "Coverage"
		legend --> :topright
		label --> method_names[2]
		gmodel_df.target_location, gmodel_df.coverage
	end 	
end