using JLD2
using MCTS
using ParticleFilters
using Plots
using HDF5
using POMDPs
include("minex_definition.jl")
include("generative_ME_belief.jl")
include("voi_policy.jl")
initialize_DGM_python("/home/acorso/Workspace/DeepGenerativeModelsCCS")

# Load all of the sample states 
s_all = imresize(h5read("planning/data/ore_maps.hdf5", "X"), (32,32))

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

# Setup the particle filters
Nparticles=10000
up_particle_loose = BootstrapFilter(m_loose, Nparticles)
b0_particle_loose = ParticleCollection(particle_set(Nparticles))

up_particle_mid = BootstrapFilter(m_mid, Nparticles)
b0_particle_mid = ParticleCollection(particle_set(Nparticles))

up_particle_tight = BootstrapFilter(m_tight, Nparticles)
b0_particle_tight = ParticleCollection(particle_set(Nparticles))

# Load the DGM based belief updaters
up_ddpm250, b0_ddpm250 = gen_DGM_belief("planning/models/ddpm_ore_maps_250.yaml", "planning/models/ddpm250.ckpt")
up_ddpm500, b0_ddpm500 = gen_DGM_belief("planning/models/ddpm_ore_maps_500.yaml", "planning/models/ddpm500.ckpt")
up_conv1, b0_conv1 = gen_DGM_belief("planning/models/config_conv.yaml", "planning/models/halfinject_conv.ckpt")
up_conv8, b0_conv8 = gen_DGM_belief("planning/models/config_conv8.yaml", "planning/models/halfinject_conv8.ckpt")

belief_representations = [
    ["Particle-Loose", up_particle_loose, b0_particle_loose],
    ["Particle-Mid", up_particle_mid, b0_particle_mid],
    ["Particle-Tight", up_particle_tight, b0_particle_tight],
    ["DDPM250", up_ddpm250, b0_ddpm250],
    ["CONV8", up_conv8, b0_conv8]
]

# Calibrating the models types:
# all_samples = rand(b0_ddpm250, 10)
# rewards1 = [extraction_reward(m_loose, s) for s in all_samples]
# mean(rewards1)

# particle_rewards = [extraction_reward(m_loose, s) for s in b0_particle_loose.particles]
# mean(particle_rewards)

# histogram(particle_rewards, alpha=0.3)
# histogram!(rewards1, alpha=0.3)

# Load in the pomcpow results
results = JLD2.load("planning/results/results_DDPM250_VOI.jld2")["results"]

hist = results[3] #Get a single history

plot_map(hist[1].s, [], [])
savefig("gt.pdf")

er = extraction_reward(m_loose, hist[3].s)
as = [h.a for h in hist]
os = [h.o for h in hist]

cur_as = []
cur_os = []

Nsamples = 100
for i=1:length(as)
    println("iteration: $i")
    all_plots = []

    for (j, (name, up, b)) in enumerate(belief_representations)
        s = rand(b, Nsamples)
        push!(all_plots, plot_samples(s, cur_as, cur_os, er, name=name))
        belief_representations[j][3] = POMDPs.update(up, b, as[i], os[i])
    end

    plot(all_plots..., layout=(1,length(all_plots)), size=(600*length(all_plots), 400*length(belief_representations)))
    savefig("belief_comparison_index_$(i-1).pdf")

    push!(cur_as, as[i])
    push!(cur_os, os[i])
end
