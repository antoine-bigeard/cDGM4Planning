using JLD2
using MCTS
using ParticleFilters
using Plots
using HDF5
using POMDPs
include("minex_definition.jl")
include("generative_ME_belief.jl")
initialize_DGM_python("/home/acorso/Workspace/DeepGenerativeModelsCCS")
input_size=(50,50)

# Load all of the sample states 
s_all = h5read("planning/data/ore_maps.hdf5", "X")

# Load the trials states
Ntrials = 100
s0_trial = [s_all[:,:,i] for i in 1:Ntrials]

# Function to load the particle set if needed
function particle_set(Nparticles)
    Random.seed!(0) # Particle set consistency
    indices = shuffle(Ntrials:size(s_all, 3))[1:Nparticles]
    return [s_all[:,:,i] for i in indices]
end

# Function to load the DGM belief and updater
function gen_DGM_belief(config, checkpoint)
    up = GenerativeMEBeliefUpdater(config, checkpoint, MinExPOMDP(), input_size)
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
    ers = [extraction_reward(m_loose, samples[:,:,i]) for i=1:size(samples, 3)]
    ret_plot = histogram(ers, bins=-150:10:200, xlabel="Extraction Rewards", title=name)
    vline!([er_gt], label="Ground Truth")

    # Get average absolute deviation to the observation points
    devs = [avg_abs_dev(samples[:,:,i], as, os) for i=1:size(samples, 3)]
    dev_plot = histogram(devs, bins=0:0.01:0.5, xlabel="Mean Abs Deviations")

    example_images = [plot_map(samples[:,:,i], as, os) for i=1:Nexamples]
    plot(ret_plot, dev_plot, example_images..., layout=(Nexamples+2, 1), size=(600, 400*(Nexamples+2)))
end

# Construct the POMDP
m_loose = MinExPOMDP(σ_abc=0.1)
m_tight = MinExPOMDP(σ_abc=0.01)

# Setup the particle filters
Nparticles=10000
up_particle_loose = BootstrapFilter(m_loose, Nparticles)
b0_particle_loose = ParticleCollection(particle_set(Nparticles))

up_particle_tight = BootstrapFilter(m_tight, Nparticles)
b0_particle_tight = ParticleCollection(particle_set(Nparticles))

# Load the DGM based belief updaters
up_ddpm250, b0_ddpm250 = gen_DGM_belief("planning/models/ddpm_ore_maps_250.yaml", "planning/models/ddpm250.ckpt")
up_ddpm500, b0_ddpm500 = gen_DGM_belief("planning/models/ddpm_ore_maps_500.yaml", "planning/models/ddpm500.ckpt")
up_conv1, b0_conv1 = gen_DGM_belief("planning/models/config_conv.yaml", "planning/models/halfinject_conv.ckpt")
up_conv8, b0_conv8 = gen_DGM_belief("planning/models/config_conv8.yaml", "planning/models/halfinject_conv8.ckpt")

# Load in the pomcpow results
results = JLD2.load("planning/results/results_POMCPOW.jld2")["results"]

hist = results[2] #Get a single history

er = extraction_reward(m_loose, hist[1].s)
as = [h.a for h in hist]
os = [h.o for h in hist]

cur_as = []
cur_os = []

for i=1:length(as)
    println("iteration: $i")
    all_plots = []

    s1 = cat([rand(b0_particle_loose) for i=1:100]..., dims=3)
    push!(all_plots, plot_samples(s1, cur_as, cur_os, er, name="Particles - Loose"))

    s2 = cat([rand(b0_particle_tight) for i=1:100]..., dims=3)
    push!(all_plots, plot_samples(s2, cur_as, cur_os, er, name="Particles - Tight"))

    s3 = rand(Random.GLOBAL_RNG, b0_ddpm250, Nsamples=100)
    push!(all_plots, plot_samples(s3, cur_as, cur_os, er, name="DDPM250"))

    s4 = rand(Random.GLOBAL_RNG, b0_ddpm500, Nsamples=100)
    push!(all_plots, plot_samples(s4, cur_as, cur_os, er, name="DDPM500"))

    s5 = rand(Random.GLOBAL_RNG, b0_conv1, Nsamples=100)
    push!(all_plots, plot_samples(s5, cur_as, cur_os, er, name="CONV1"))

    s6 = rand(Random.GLOBAL_RNG, b0_conv8, Nsamples=100)
    push!(all_plots, plot_samples(s6, cur_as, cur_os, er, name="CONV8"))

    plot(all_plots..., layout=(1,length(all_plots)), size=(600*6, 400*6))
    savefig("belief_comparison_index_$(i-1).pdf")

    ## Update the beliefs
    b0_particle_loose = POMDPs.update(up_particle_loose, b0_particle_loose, as[i], os[i])
    b0_particle_tight = POMDPs.update(up_particle_tight, b0_particle_tight, as[i], os[i])
    b0_ddpm250 = POMDPs.update(up_ddpm250, b0_ddpm250, as[i], os[i])
    b0_ddpm500 = POMDPs.update(up_ddpm500, b0_ddpm500, as[i], os[i])
    b0_conv1 = POMDPs.update(up_conv1, b0_conv1, as[i], os[i])
    b0_conv8 = POMDPs.update(up_conv8, b0_conv8, as[i], os[i])

    push!(cur_as, as[i])
    push!(cur_os, os[i])
end
