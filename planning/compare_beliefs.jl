using JLD2
using MCTS
using ParticleFilters
using Plots
using HDF5
using POMDPs
using Distributions
include("minex_definition.jl")
include("generative_ME_belief.jl")
include("voi_policy.jl")
initialize_DGM_python("/home/acorso/Workspace/DeepGenerativeModelsCCS")

# Load all of the sample states 
s_all = imresize(h5read("planning/data/ore_maps.hdf5", "X"), (32,32))

# Load the trials states
Ntrials = 100
s0_trial = [MinExState(s_all[:,:,i]) for i in 1:Ntrials]

# Function to load the particle set if needed
function particle_set(Nparticles)
    Random.seed!(0) # Particle set consistency
    indices = shuffle(Ntrials:size(s_all, 3))[1:Nparticles]
    return [MinExState(s_all[:,:,i]) for i in indices]
end

# Function to load the DGM belief and updater
function gen_DGM_belief(config, checkpoint)
    up = GenerativeMEBeliefUpdater(config, checkpoint, MinExPOMDP(), (32,32)) #TODO: remove input size
    b0 = initialize_belief(up, nothing)
    up, b0
end

# function to plot map with as and obs
function plot_map(s, as, os)
    p=heatmap(s', clims=(0,1), cbar=false)
    for (a, o) in zip(as, os)
        scatter!([a[1]], [a[2]], marker_z=o, markerstrokecolor=:green, markersize=5, label="")
    end
    p
end

function avg_abs_dev(sample, as, os)
    dev = 0
    for (a, o) in zip(as, os)
        dev += abs(sample[a...] - o) / length(as)
    end
    return dev
end

# Function for plotting belief stats
function plot_samples(samples, as, os, er_gt; Nexamples=4, name)
    # Get the distribution of extractionr rewards and compare to ground truth
    ers = [extraction_reward(m_loose, s) for s in samples]
    ret_plot = histogram(ers, bins=-150:10:200, xlabel="Extraction Rewards", title=name)
    vline!([er_gt], label="Ground Truth")

    # Get average absolute deviation to the observation points
    devs = [avg_abs_dev(s, as, os) for s in samples]
    dev_plot = histogram(devs, bins=0:0.01:0.5, xlabel="Mean Abs Deviations")

    example_images = [plot_map(s, as, os) for s in samples[1:Nexamples]]
    plot(ret_plot, dev_plot, example_images..., layout=(Nexamples+2, 1), size=(600, 400*(Nexamples+2)))
end

# Construct the POMDP
m_loose = MinExPOMDP(σ_abc=0.1)
m_mid = MinExPOMDP(σ_abc=0.05)
m_tight = MinExPOMDP(σ_abc=0.01)

####################################################################
## Calibrating the models types:
# up_ddpm250, b0_ddpm250 = gen_DGM_belief("planning/models/ddpm_ore_maps_250.yaml", "planning/models/ddpm250.ckpt")

# all_samples = rand(b0_ddpm250, 10)
# rewards1 = [extraction_reward(m_loose, s) for s in all_samples]
# mean(rewards1)

# Nparticles=10000
# b0 = ParticleCollection(particle_set(Nparticles))
# m = MinExPOMDP(extraction_cost = 52)
# particle_rewards = [extraction_reward(m, s) for s in b0.particles]
# mean(particle_rewards)

# histogram(particle_rewards, alpha=0.3, label="Particles")
# histogram!(rewards1, alpha=0.3, label="DDPM250")

####################################################################
## Computing "best case" rewards
assumed_accuracy = 0.9
average_drills = 20
rs = [extraction_reward(m_loose, s) for s in s0_trial]
best_return = mean([-.1*average_drills + mean([rand() < assumed_accuracy ? max(0,r) : min(r, 0) for r in rs]) for i=1:100])
best_return
best_return

####################################################################
# ## Comparing belief representations over a trajcetory
# # Load in the pomcpow results
# results = JLD2.load("planning/results/results_DDPM250_tree_search.jld2")["results"]

# for (state_i, hist) in enumerate(results)
#     # Setup the particle filters
#     Nparticles=10000
#     up_particle_loose = BootstrapFilter(m_loose, Nparticles)
#     b0_particle_loose = ParticleCollection(particle_set(Nparticles))

#     up_particle_mid = BootstrapFilter(m_mid, Nparticles)
#     b0_particle_mid = ParticleCollection(particle_set(Nparticles))

#     up_particle_tight = BootstrapFilter(m_tight, Nparticles)
#     b0_particle_tight = ParticleCollection(particle_set(Nparticles))

#     # Load the DGM based belief updaters
#     up_ddpm250, b0_ddpm250 = gen_DGM_belief("planning/models/ddpm_ore_maps_250.yaml", "planning/models/ddpm250.ckpt")
#     up_ddpm500, b0_ddpm500 = gen_DGM_belief("planning/models/ddpm_ore_maps_500.yaml", "planning/models/ddpm500.ckpt")
#     up_conv1, b0_conv1 = gen_DGM_belief("planning/models/config_conv.yaml", "planning/models/halfinject_conv.ckpt")
#     up_conv8, b0_conv8 = gen_DGM_belief("planning/models/config_conv8.yaml", "planning/models/halfinject_conv8.ckpt")

#     # All belief representations
#     belief_representations = [
#         ["Particle-Loose", up_particle_loose, b0_particle_loose],
#         ["Particle-Mid", up_particle_mid, b0_particle_mid],
#         ["Particle-Tight", up_particle_tight, b0_particle_tight],
#         ["DDPM250", up_ddpm250, b0_ddpm250],
#         ["CONV8", up_conv8, b0_conv8]
#     ]

#     # Set up the directory
#     dir = "case_$(state_i)"
#     try mkdir(dir) catch end

#     plot_map(hist[1].s, [], [])
#     savefig("$dir/gt.pdf")

#     er = extraction_reward(m_loose, hist[1].s)
#     as = [h.a for h in hist]
#     os = [h.o for h in hist]

#     cur_as = []
#     cur_os = []

#     Nsamples = 100
#     for i=1:length(as)
#         println("iteration: $i")
#         all_plots = []

#         for (j, (name, up, b)) in enumerate(belief_representations)
#             s = rand(b, Nsamples)
#             push!(all_plots, plot_samples(s, cur_as, cur_os, er, name=name))
#             belief_representations[j][3] = POMDPs.update(up, b, as[i], os[i])
#         end

#         plot(all_plots..., layout=(1,length(all_plots)), size=(600*length(all_plots), 400*length(belief_representations)))
#         savefig("$dir/belief_comparison_index_$(i-1).pdf")

#         push!(cur_as, as[i])
#         push!(cur_os, os[i])
#     end
# end

# ####################################################################
# ## Check accuracy of mining decision vs number of observations
# Nparticles=10000
# up_particle_loose = BootstrapFilter(m_loose, Nparticles)
# b0_particle_loose = ParticleCollection(particle_set(Nparticles))

# up_particle_mid = BootstrapFilter(m_mid, Nparticles)
# b0_particle_mid = ParticleCollection(particle_set(Nparticles))

# up_particle_tight = BootstrapFilter(m_tight, Nparticles)
# b0_particle_tight = ParticleCollection(particle_set(Nparticles))

# # Load the DGM based belief updaters
# up_ddpm250, b0_ddpm250 = gen_DGM_belief("planning/models/ddpm_ore_maps_250.yaml", "planning/models/ddpm250.ckpt")
# up_ddpm500, b0_ddpm500 = gen_DGM_belief("planning/models/ddpm_ore_maps_500.yaml", "planning/models/ddpm500.ckpt")
# up_conv1, b0_conv1 = gen_DGM_belief("planning/models/config_conv.yaml", "planning/models/halfinject_conv.ckpt")
# up_conv8, b0_conv8 = gen_DGM_belief("planning/models/config_conv8.yaml", "planning/models/halfinject_conv8.ckpt")

# belief_representations = [
#         ["Particle-Loose", up_particle_loose, b0_particle_loose],
#         ["Particle-Mid", up_particle_mid, b0_particle_mid],
#         ["Particle-Tight", up_particle_tight, b0_particle_tight],
#         ["DDPM250", up_ddpm250, b0_ddpm250],
#         ["DDPM500", up_ddpm500, b0_ddpm500],
#         ["CONV1", up_conv1, b0_conv1],
#         ["CONV8", up_conv8, b0_conv8]
#     ]

# n_drills_list = collect(length(m_loose.drill_locations):-3:1)
# Nsamps = 50
# all_decision_accuracies = []
# all_oremass_errors = []
# all_pdfs = []

# function plot_progress(all_vals, ylabel)
#     p = plot(xlabel="Number of drills", ylabel=ylabel, title="$ylabel vs number of observations")
#     for name in keys(all_vals[1])
#         vals = [a[name] for a in all_vals]
#         plot!(n_drills_list[1:length(vals)], vals, label=name, marker=true)
#     end
#     p
# end

# for n_drills in n_drills_list
#     decision_accuracies = Dict()
#     oremass_errors = Dict()
#     pdfs = Dict()
#     for (i,s) in enumerate(s0_trial[1:Nsamps])
#         # Get the ground truth extraction reward
#         r_truth = extraction_reward(m_loose, s)

#         # Get a random set of drill locations
#         as = sample(m_loose.drill_locations, n_drills; replace=false)

#         # Get the corresponding observations
#         _, os, _ = gen(m_loose, s, as, Random.GLOBAL_RNG)

#         println("==> N_drills: ", n_drills, "   trial: ", i, " len as: ", length(as), " len obs: ", length(os))


#         # Loop through the belief representations and figure out if they would drill given the info
#         for (name, up, b) in belief_representations
#             println("Belief type: ", name)
#             bp = POMDPs.update(up, b, as, os) # Update the belief
#             s_samples = rand(bp, 100) # Get 100 samples
#             rbs = [extraction_reward(m_loose, si) for si in s_samples]

#             rb = mean(rbs) # Get the mean extraction reward according to the updated belief

#             # If the belief is "correct" about extracting, add 1/Ntrials to the accuracy
#             decision_accuracies[name] = get(decision_accuracies, name, 0) + Float64(sign(r_truth) == sign(rb))/Nsamps
#             oremass_errors[name] = get(oremass_errors, name, 0) + abs(r_truth - rb)/Nsamps
#             pdfs[name] = get(pdfs, name, 0) + pdf(Normal(rb, std(rbs)), r_truth)/Nsamps
#         end
#     end
#     push!(all_decision_accuracies, decision_accuracies)
#     push!(all_oremass_errors, oremass_errors)
#     push!(all_pdfs, pdfs)

#     p1 = plot_progress(all_decision_accuracies, "Decision Accuracy")
#     p2 = plot_progress(all_oremass_errors, "Ore Mass Error")
#     p3 = plot_progress(all_pdfs, "pdf of GT")
#     p = plot(p1, p2, p3, layout=(1, 3), size=(1800, 400))
#     display(p)
# end

# savefig("belief_comparisons.pdf")

# JLD2.save("belief_comparisons_data.jld2", Dict("decicison_accuracy" => all_decision_accuracies, "oremass_errors" => all_oremass_errors, "pdfs" => all_pdfs))



# ## Plotting some random results 
# s = s0_trial[3]
# up, b = up_ddpm250, b0_ddpm250

# r_truth = extraction_reward(m_loose, s)

# # Get a random set of drill locations
# as = sample(m_loose.drill_locations, 36; replace=false)

# # Get the corresponding observations
# _, os, _ = gen(m_loose, s, as, Random.GLOBAL_RNG)

# bp = POMDPs.update(up, b, as, os) # Update the belief
# s_samples = rand(bp, 100) # Get 100 samples
# rb = mean([extraction_reward(m_loose, si) for si in s_samples]) # Get the mean extraction reward according to the updated belief

# plot_map(s, as, os)