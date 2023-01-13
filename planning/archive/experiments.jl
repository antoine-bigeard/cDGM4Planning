include("minex_definition.jl")
include("generative_ME_belief.jl")
using ParticleFilters
using HDF5
using Random
using Plots
using MCTS
using JLD2

## Parameters
Ntrials = 1
Nparticles = 10000
tree_queries = 1 # Tree queries for the planner
exploration_constant=100.0
a_widening = (;k_action = 2.0, alpha_action = 0.25,)
o_widening = (;k_observation=2.0,alpha_observation=0.1,)

                

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

# # Create the updater and initial belief
# b0 = ParticleCollection(particle_states)
# up = BootstrapFilter(m, Nparticles)

# # Get random policy results
# random_policy = RandomPolicy(m)

# solver_pomcpow = POMDPs.solve(POMCPOWSolver(next_action=MinExActionSampler(), 
#                                              estimate_value=0;
#                                              criterion=POMCPOW.MaxUCB(exploration_constant),
#                                              tree_queries, 
#                                              a_widening...,
#                                              o_widening...), m)

# rand_results = []
# pomcpow_results = []
# for (i, s0) in enumerate(s0_trial)
#     println("Trial $i")
#     println("Random Policy")
#     push!(rand_results, simulate(HistoryRecorder(), m, random_policy, up, b0, s0))
#     JLD2.save("planning/results/results_random.jld2", Dict("results" => rand_results))

#     println("POMCPOW")
#     push!(pomcpow_results, simulate(HistoryRecorder(), m, solver_pomcpow, up, b0, s0))
#     JLD2.save("planning/results/results_POMCPOW.jld2", Dict("results" => pomcpow_results))
    
# end
DGM = "/home/acorso/Workspace/DeepGenerativeModelsCCS"
initialize_DGM_python(DGM)

input_size=(50,50)
function gen_bmdp(config, checkpoint)
    up = GenerativeMEBeliefUpdater(config, checkpoint, m, input_size)
    b0 = initialize_belief(up, nothing)
    return up, b0, GenerativeBeliefMDP{typeof(m), typeof(up), typeof(b0), actiontype(m)}(m, up)
end

configs = [
    "planning/models/ddpm_ore_maps_250.yaml",
    "planning/models/ddpm_ore_maps_500.yaml",
    "planning/models/config_conv.yaml",
    "planning/models/config_conv8.yaml"
]

models = [
    "planning/models/ddpm250.ckpt",
    "planning/models/ddpm500.ckpt",
    "planning/models/halfinject_conv.ckpt",
    "planning/models/halfinject_conv8.ckpt"
]

names = ["DDPM250", "DDPM500", "CONV1", "CONV8"]

for (config, model, name) in zip(configs, models, names)
    println("Name: ", name)
    up, b0, bmdp = gen_bmdp(config, model)
    solver = solve(DPWSolver(n_iterations=tree_queries; exploration_constant, a_widening...), bmdp)
    results = []
    for (i,s0) in enumerate(s0_trial)
        push!(results, simulate(HistoryRecorder(), m, solver, up, b0, s0))
        JLD2.save("planning/results/results_$name.jld2", Dict("results" => results))
    end
end 

