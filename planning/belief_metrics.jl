using HDF5
using ParticleFilters
using JLD2
using Plots, Measures; pgfplotsx()
import DataStructures: OrderedDict
using Printf
using LaTeXStrings
include("minex_definition.jl")
include("generative_ME_belief.jl")
initialize_DGM_python("/home/acorso/Workspace/DeepGenerativeModelsCCS")

# Function to load the DGM belief and updater
function gen_DGM_belief(config, checkpoint)
    up = GenerativeMEBeliefUpdater(config, checkpoint, MinExPOMDP(), (32,32)) #TODO: remove input size
    b0 = initialize_belief(up, nothing)
    up, b0
end

save_file = "all_results.jld2"
m = MinExPOMDP()
Nval = 100 #10 #100 # Number of ore maps to evaluate the belief representations on
Nsamples = 500 #100 #500 # Number of samples to draw from the belief for each ore map
Naction_step = 3 #6 #3 # number of actions to step between on each evaluation. 

# Load in the train and test data
train_ore_maps = imresize(h5read("planning/data/ore_maps.hdf5", "X"), (32,32))
test_ore_maps = imresize(h5read("planning/data/test_ore_maps.hdf5", "X"), (32,32))

# setup the test set and the correspoinding set of drill locations (same for all beliefs)
test_maps = [MinExState(test_ore_maps[:,:,i]) for i in 1:Nval]
as_per_s = [[sample(m.drill_locations, i; replace=false) for _=1:Nval] for i=0:Naction_step:length(m.drill_locations)]


## Load belief representations
belief_representations = Dict()

# Load all of the particle filter representations
particle_counts = [1_000, 10_000, 100_000]
abc_params = [0.1, 0.05, 0.01]

for pc in particle_counts, abc in abc_params
    function load_pf()
        b = ParticleCollection([MinExState(train_ore_maps[:,:,i]) for i in 1:pc])
        up = BootstrapFilter(MinExPOMDP(σ_abc=abc), pc)
        up, b
    end
    belief_representations["Particle-$(pc)k-$abc"] = () -> load_pf()
end

for name in readdir("planning/models"; join=false)
    belief_representations[name] = () -> gen_DGM_belief("planning/models/$name/$name.yaml", "planning/models/$name/$name.ckpt")
end

## Process the belief representations
results = Dict()
# results = JLD2.load("coarse_all_belief_comparisons.jld2")
fulltime = @elapsed for (name, fn) in belief_representations
    println("going for: ", name)
    up, b = fn() # load the updater and belief
    results[name] = Dict("Number of Actions" => [], "Conditioning Error"=>[], "Min L2"=>[], "Time ($Nsamples Samples)"=>[], "Decision Accuracy"=>[], "Ore Value Error"=>[], "Probability Density of Ore Value"=>[])
    for as in as_per_s
        # Set up the arrays to store each metric
        update_and_inference_times = []
        minL2_errors = []
        minL2_sample = []
        conditioning_errors = []
        decision_accuracies = []
        oremass_errors = []
        logpdfs = []

        for i=1:Nval
            println("name: $name, num actions: $(length(as[1])), val index: ", i)

            # Get the observation for this state and action 
            s = test_maps[i]
            r_truth = extraction_reward(m, s)
            a = as[i]
            _, o, _ = gen(m, s, a, Random.GLOBAL_RNG)

            # update the belief and sample from the posterior
            time = @elapsed begin
                bp = POMDPs.update(up, b, a, o) # Update the belief
                s_samples = rand(bp, Nsamples)
            end
            push!(update_and_inference_times, time)

            ## Compute the task-independent metrics
            L2errors = [norm(s.ore .- si.ore) for si in s_samples]
            min_samp_index = argmin(L2errors)
            push!(minL2_sample, s_samples[min_samp_index])
            push!(minL2_errors, L2errors[min_samp_index])

            avg_cond_errors = [mean(abs.(o .- gen(m, si, a, Random.GLOBAL_RNG)[2])) for si in s_samples]
            push!(conditioning_errors, mean(avg_cond_errors))

            ## Compute the task-dependent metrics
            rbs = [extraction_reward(m, si) for si in s_samples] # extraction reward for each sample
            rb = mean(rbs) # average predicted extraction reward
            push!(decision_accuracies, sign(r_truth) == sign(rb))
            push!(oremass_errors, abs(r_truth - rb))
            push!(logpdfs, logpdf(Normal(rb, std(rbs)), r_truth))
        end
        push!(results[name]["Number of Actions"], length(as[1])) # record the number of drill locations
        push!(results[name]["Conditioning Error"], conditioning_errors)
        push!(results[name]["Min L2"], minL2_errors)
        push!(results[name]["Time ($Nsamples Samples)"], update_and_inference_times)
        push!(results[name]["Decision Accuracy"], decision_accuracies)
        push!(results[name]["Ore Value Error"], oremass_errors)
        push!(results[name]["Probability Density of Ore Value"], logpdfs)
        
        JLD2.save(save_file, results)

        # p = plot_metrics(results, ["Conditioning Error", "Min L2", "Time ($Nsamples Samples)", "Decision Accuracy", "Ore Value Error", "Probability Density of Ore Value"], layout=(2,3))
        # savefig(p, "all_metrics.pdf")
    end
end


## Process the results
function gen_table(results, names, metrics, headers)
    for (i, h) in enumerate(headers)
        if i==1
            print(h)
        else
            print(" & $h")
        end
    end
    println("\\\\")
    println("\\midrule")

    for ((_, res), name) in zip(results, names)
        row = "$name"
        for m in metrics
            if m == "Probability Density of Ore Value"
                vals = collect(Iterators.flatten(res[m]))
                vals[isinf.(vals)] .= -Inf
                vals = exp.(vals)
            elseif m=="Conditioning Error"
                vals = collect(Iterators.flatten(res[m][2:end]))
            else
                vals = Iterators.flatten(res[m])
            end
            meanval = mean(vals)
            stdval = std(vals)
            el = @sprintf("\$%.2f (%.2f)\$", meanval, stdval)
            row = "$row & $el"
        end
        row = "$row\\\\"
        println(row)
    end
end

function plot_metrics(results, 
    metrics; 
    layout, 
    names=keys(results),
    colors=collect(1:length(names)), 
    linestyles=fill(:solid, length(names)),
    xlabels=fill("Number of Actions", prod(layout)),
    ylabels=metrics,
    titles=metrics,
    kwargs...
    )
    plots = []
    for i=1:prod(layout)
        legend = i==5 ? :bottom : false
        push!(plots, plot_metric(results, metrics[i], xlabel=xlabels[i], ylabel=ylabels[i], title=titles[i]; names, legend, colors, linestyles, kwargs...))
    end
    plot(plots...; layout)
end

function plot_metric(results, metric; ylabel=metric, title=metric, names=keys(results), linestyles, colors, kwargs...)
    println("metrics: ", metric, " kwargs: ", kwargs)
    p = plot(palette=:Set2_3; ylabel, title, legend_column=3, kwargs...)
    for ((_, res), name, linestyle, color) in zip(results, names, linestyles, colors)
        if metric == "Probability Density of Ore Value"
            vals = []
            for arr in res[metric]
                arr[isinf.(arr)] .= -Inf
                arr = exp.(arr)
                push!(vals, mean(arr))
            end
        else
            vals = mean.(res[metric])
        end
        plot!(p, res["Number of Actions"], vals, label=name; linestyle, color)
    end
    p
end

all_results = JLD2.load("stylegan_coarse.jld2")
all_keys = collect(keys(all_results))
Nsamples=100

metrics = ["Min L2", "Conditioning Error", "Ore Value Error", "Decision Accuracy",  "Probability Density of Ore Value", "Time ($Nsamples Samples)"]
headers = ["\\makecell{Min\\\\\$L_{2}\$(\$\\downarrow\$)}", "\\makecell{Conditioning\\\\Error(\$\\downarrow\$)}", "\\makecell{Ore Value\\\\Error(\$\\downarrow\$)}", "\\makecell{Decision\\\\Accuracy(\$\\uparrow\$)}", "\\makecell{Prob. of\\\\Ore Value(\$\\uparrow\$)}", "\\makecell{Time(\$\\downarrow\$)}"]

particle_models_in_order = [
    "Particle-1000k-0.1", 
    "Particle-1000k-0.05", 
    "Particle-1000k-0.01",
    "Particle-10000k-0.1", 
    "Particle-10000k-0.05", 
    "Particle-10000k-0.01",    
    "Particle-100000k-0.1", 
    "Particle-100000k-0.05", 
    "Particle-100000k-0.01"
]
particle_names = [
    "\$1000\$ & \$0.1\$",
    "& \$0.05\$",
    "& \$0.01\$",
    "\$10000\$ & \$0.1\$",
    "& \$0.05\$",
    "& \$0.01\$",
    "\$100000\$ & \$0.1\$",
    "& \$0.05\$",
    "& \$0.01\$",
]
particle_headers = ["\$N_{\\rm Particles}\$ & \$\\sigma_{\\rm ABC}\$"]

ddpm_models_in_order = [
    "ddpm_ore_maps_250_small", 
    "ddpm_ore_maps_500_small",
    "ddpm_ore_maps_250_medium", 
    "ddpm_ore_maps_500_medium", 
    "ddpm_ore_maps_100", 
    "ddpm_ore_maps_150", 
    "ddpm_ore_maps_200", 
    "ddpm_ore_maps_250", 
    "ddpm_ore_maps_500", 
    "ddpm_ore_maps_1000", 
]
ddpm_names = [
    "Small & \$250\$",
    "& \$500\$",
    "Medium & \$250\$",
    "& \$500\$",
    "Large & \$100\$",
    "& \$150\$",
    "& \$200\$",
    "& \$250\$",
    "& \$500\$",
    "& \$1000\$",
]
ddpm_headers = ["\\makecell{Model\\\\Size} & \$N_{\\rm Iterations}\$"]


gan_models_in_order = [
    "gan_1inject_conv8_latent64_w",
    "gan_1inject_conv8_latent128_w",
    "gan_halfinject_conv2_latent32_w",
    "gan_halfinject_conv2_latent64_w",
    "gan_halfinject_conv2_latent128_w",
    "gan_halfinject_conv8_latent128_w",
    "gan_fullinject_conv2_latent64_w",
    "gan_fullinject_conv2_latent128_w",
    "gan_fullinject_conv8_latent64_w",
    "gan_fullinject_conv8_latent128_w",
    "gan_fullinject_conv8_latent128_ce",
    "gan_fullinject_conv8_latent128_wgp",
]
gan_names = [
    "1st Layer & (\$8\$, \$64\$, W)",
    "& (\$8\$, \$128\$, W)",
    "Half Layers & (\$2\$, \$32\$, W)",
    "& (\$2\$, \$64\$, W)",
    "& (\$2\$, \$128\$, W)",
    "& (\$8\$, \$128\$, W)",
    "All Layers & (\$2\$, \$64\$, W)",
    "& (\$2\$, \$128\$, W)",
    "& (\$8\$, \$64\$, W)",
    "& (\$8\$, \$128\$, W)",
    "& (\$8\$, \$128\$, CE)",
    "& (\$8\$, \$128\$, W-GP)",
]
gan_headers = ["\\makecell{Condition\\\\Injection} & \\makecell{(Channels,\\\\Latent Dim,\\\\Loss)}"]


stylegan_models_in_order = [
  "stylegan_cond_w",
  "stylegan_cond_ce",
  "stylegan_cond_wgp_nonoise-smooth"
]
stylegan_names = [
    "W",
    "CE",
    "W-GP"
]
stylegan_headers = ["\\makecell{Loss\\\\Function}"]

## Comparisons
comparisons_in_order = [
    "Particle-10000k-0.05",
    "Particle-100000k-0.05",
    "gan_halfinject_conv2_latent32_w",
    "gan_fullinject_conv2_latent128_w",
    "ddpm_ore_maps_250",
    "ddpm_ore_maps_500",
]
comparison_names = [
    "PF (\$10\$k, \$0.05\$)",
    "PF (\$100\$k, \$0.05\$)",
    "GAN (Half, \$2\$, \$32\$, W)",
    "GAN (All, \$2\$, \$128\$, W)",
    "DDPM (Large, \$250\$)",
    "DDPM (Large, \$500\$)"
]
latex_comparison_names = [
    L"PF ($10$k, $0.05$)",
    L"PF ($100$k, $0.05$)",
    L"GAN (Half, $2$, $32$, W)",
    L"GAN (All, $2$, $128$, W)",
    L"DDPM (Large, $250$)",
    L"DDPM (Large, $500$)"
]
comparison_headers = ["\\makecell{Belief\\\\Representation}"]

## Generate tables
gen_table(OrderedDict(k=>all_results[k] for k in particle_models_in_order), particle_names, metrics, [particle_headers..., headers...])
gen_table(OrderedDict(k=>all_results[k] for k in ddpm_models_in_order), ddpm_names, metrics, [ddpm_headers..., headers...])
gen_table(OrderedDict(k=>all_results[k] for k in gan_models_in_order), gan_names, metrics, [gan_headers..., headers...])
gen_table(OrderedDict(k=>all_results[k] for k in stylegan_models_in_order), stylegan_names, metrics, [stylegan_headers..., headers...])
gen_table(OrderedDict(k=>all_results[k] for k in comparisons_in_order), comparison_names, metrics, [comparison_headers..., headers...])

# Plot Metrics
plot_metrics(
    OrderedDict(k=>all_results[k] for k in comparisons_in_order), 
    metrics, 
    layout=(2,3), 
    xlabels=["","","","Number of Actions", "Number of Actions", "Number of Actions"],
    ylabels=[L"$L_2$", "Absolute Error", "Absolute Error", "Accuracy", "Probability Density", "Time (seconds)"],
    titles= [ L"Min $L_2$", "Conditioning Error", "Ore Value Error", "Decision Accuracy", "Probability Density of Ore Value", "Time (100 Samples)"],
    names=latex_comparison_names,
    colors = [1,1,2,2,3,3],
    linestyles=[:solid, :dash, :solid, :dash, :solid, :dash],
    xlabelfontsize=6,
    ylabelfontsize=6,
    titlefontsize=8,
    legendfontsize=6
    )
savefig("comparisons.tex")
# NOTE: For fixing the label spacing of the plot: Remove everything in the x- and y-label style section up to font, changed height and y-shift. Changed legend location 


## Generate samples
function plot_map(s, as, os; ylabel=false)
    p=heatmap(s', clims=(0,1), cbar=false, yticks=false, xticks=false, xaxis=false, yaxis=false, grid=false, size=(100,100), ylabel=ylabel, ylabelfontsize=8,)
    scatter!([], [], markerstrokecolor=:white, markersize=1, label="")
    for (a, o) in zip(as, os)
        scatter!([a[1]], [a[2]], marker_z=o, markerstrokecolor=:white, markersize=3, label="")
    end
    p
end

Nsamples = 500
column_plots = []
best_samples = Dict()
for i=[1, 2, 4, 8, 13]
    val_i = rand(1:100)
    a = as_per_s[i][val_i]
    s = test_maps[val_i]
    _, o, _ = gen(m, s, a, Random.GLOBAL_RNG)

    one_column = [plot_map(s.ore, a, o, ylabel=i==1 ? "Ground Truth" : "")]
    for (belief_type, name) in zip(comparisons_in_order[[1,3,6]], latex_comparison_names[[1,3,6]])
        fn = belief_representations[belief_type]
        up, b = fn() 
        bp = POMDPs.update(up, b, a, o) # Update the belief
        s_samples = rand(bp, Nsamples)
        L2errors = [norm(s.ore .- si.ore) for si in s_samples]
        sbest = s_samples[argmin(L2errors)]
        best_samples["$(i)_$(val_i)"] = sbest
        p = plot_map(sbest.ore, a, o, ylabel=i==1 ? name : "")
        push!(one_column, p)
    end
    push!(column_plots, plot(one_column..., layout=(length(comparisons_in_order)+1,1)))
end
JLD2.save("best_samples.jld2", best_samples)
plot(column_plots..., layout=(1, length(column_plots)), size=(800,1000), margin=-10mm)

savefig("samples.pdf")