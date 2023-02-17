# Run this file with a command such as
# export CUDA_VISIBLE_DEVICES=1
#julia1.8 planning/run_experiment.jl planning/configs/random_policy.yaml

using YAML
using HDF5
using JLD2
using Random
using MCTS
using ParticleFilters
using Images
include("minex_definition.jl")
include("voi_policy.jl")
include("generative_ME_belief.jl")

# load the param file
config = YAML.load_file(ARGS[1])
name = config["name"] 

if occursin("DGM", config["trial_type"])
    # Load code for generative models
    initialize_DGM_python(config["DGM_path"])
end

# Construct the POMDP
σ_abc = haskey(config, "ABC_param") ? config["ABC_param"] : 0.1
m = MinExPOMDP(;σ_abc, drill_locations = [(i,j) for i=3:7:31 for j=3:7:31])

# Load the ore maps
s_all = imresize(h5read("planning/data/ore_maps.hdf5", "X"), (32,32))
s_test = imresize(h5read("planning/data/test_ore_maps.hdf5", "X"), (32,32))

# Load the trials states
Ntrials = config["Ntrials"]
s0_trial = [MinExState(s_test[:,:,i]) for i in 1:Ntrials]

# Function to load the particle set if needed
function particle_set(Nparticles)
    Random.seed!(0) # Particle set consistency
    indices = shuffle(1:size(s_all, 3))[1:Nparticles]
    return [MinExState(s_all[:,:,i]) for i in indices]
end

# Setup the belief, updater and policy for each type of trial
if config["trial_type"] == "random"
    up = NothingUpdater()
    b0 = initialize_belief(up, nothing)
    policy = RandomPolicy(m)
elseif config["trial_type"] == "pomcpow"
    Nparticles = config["Nparticles"]
    b0 = ParticleCollection(particle_set(Nparticles))
    up = BootstrapFilter(m, Nparticles)
    solver = POMCPOWSolver(next_action=MinExActionSampler(), 
                           estimate_value=(pomdp, s, h, steps) -> isterminal(pomdp, s) ? 0 : max(0, extraction_reward(pomdp, s)),
                           criterion=POMCPOW.MaxUCB(config["exploration_constant"]),
                           tree_queries=config["tree_queries"],
                           k_action=config["k_action"],
                           alpha_action=config["alpha_action"],
                           k_observation=config["k_observation"],
                           alpha_observation=config["alpha_observation"],
                           tree_in_info=false,
                          )
    policy = POMDPs.solve(solver, m)
elseif config["trial_type"] == "DGM_tree_search"
    up = GenerativeMEBeliefUpdater(config["model_config"], config["model_ckpt"], m, (32,32)) # TODO: remove input size
    b0 = initialize_belief(up, nothing)
    bmdp = GenerativeBeliefMDP{typeof(m), typeof(up), typeof(b0), actiontype(m)}(m, up)
    solver = DPWSolver(next_action=MinExActionSampler(), 
                       estimate_value=DGM_value_est,
                       n_iterations=config["tree_queries"],
                       exploration_constant=config["exploration_constant"],
                       k_action=config["k_action"],
                       alpha_action=config["alpha_action"],
                       k_state=config["k_state"],
                       alpha_state=config["alpha_state"],
                       tree_in_info=true,
                       )
    policy = POMDPs.solve(solver, bmdp)
elseif config["trial_type"] == "PF_VOI"
    Nparticles = config["Nparticles"]
    b0 = ParticleCollection(particle_set(Nparticles))
    up = BootstrapFilter(m, Nparticles)
    policy = VOIPolicy(m, up, config["Nobs_VOI"], config["Nsamples_est_VOI"])
elseif config["trial_type"] == "PF_VOI_Multi"
    Nparticles = config["Nparticles"]
    b0 = ParticleCollection(particle_set(Nparticles))
    up = BootstrapFilter(m, Nparticles)
    policy = VOIMultiActionPolicy(m, up, config["Nobs_VOI"], config["Nsamples_est_VOI"], config["N_mc_actions_VOI"])
elseif config["trial_type"] == "DGM_VOI"
    up = GenerativeMEBeliefUpdater(config["model_config"], config["model_ckpt"], m, (32,32))
    b0 = initialize_belief(up, nothing)
    policy = VOIPolicy(m, up, config["Nobs_VOI"], config["Nsamples_est_VOI"])
elseif config["trial_type"] == "DGM_VOI_Multi"
    up = GenerativeMEBeliefUpdater(config["model_config"], config["model_ckpt"], m, (32,32))
    b0 = initialize_belief(up, nothing)
    policy = VOIMultiActionPolicy(m, up, config["Nobs_VOI"], config["Nsamples_est_VOI"], config["N_mc_actions_VOI"])
else
    error("Unrecognized trial type: ", config["trial_type"])
end

# Run the trials
results = []
for (i,s0) in enumerate(s0_trial)
    println("Running trial $i of $name...")
    push!(results, simulate(HistoryRecorder(), m, policy, up, b0, s0))

    # Save all results after every iteration
    println("Saving trial $i of $name...")
    JLD2.save("$(config["savefolder"])/results_$name.jld2", Dict("results" => results)) 
end
