include("minex_definition.jl")
include("experiment_utils.jl")
using ParticleFilters
using HDF5
using Random
using Plots

## Parameters
Ntrials = 100
Nparticles = 10000
tree_queries = 1_000 # Tree queries for the planner

tree_params = (;k_action = 2.0,
                alpha_action = 0.25,
                k_observation=2.0,
                alpha_observation=0.1,
                criterion=POMCPOW.MaxUCB(100.0))


# Construct the POMDP
m = MinExPOMDP()

# Load the ore maps
s_all = h5read("planning/data/ore_maps.hdf5", "X")

# Load the trials states
s0_trial = [s_all[:,:,i] for i in 1:Ntrials]

# Load the particle set
Random.seed!(0) # Particle set consistency
indices = shuffle(Ntrials:size(s_all, 3))[1:Nparticles]
particle_states = [s_all[:,:,i] for i in indices]

# Create the updater and initial belief
b0 = ParticleCollection(particle_states)
up = BootstrapFilter(m, Nparticles)

# Get random policy results
random_policy = RandomPolicy(m)

pomcpow_planner = POMDPs.solve(POMCPOWSolver(next_action=MinExActionSampler(), 
                                             estimate_value=0;
                                             tree_queries, 
                                             tree_params...), m)


rand_results = []
pomcpow_results = []
for (i, s0) in enumerate(s0_trial)
    println("Trial $i")
    push!(rand_results, simulate(RolloutSimulator(), m, random_policy, up, b0, s0))
    push!(pomcpow_results, simulate(RolloutSimulator(), m, pomcpow_planner, up, b0, s0))
end

# histogram the results
histogram(rand_results, label="Random Policy", bins=-150:10:200)
histogram!(pomcpow_results, label="POMCPOW Planner", bins=-150:10:200)
