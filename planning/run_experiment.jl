# Run this file with a command such as
# export CUDA_VISIBLE_DEVICES=1
#julia1.8 planning/run_experiment.jl planning/configs/random_policy.yaml
#julia1.8 planning/run_experiment.jl planning/configs/pomcpow.yaml
#julia1.8 planning/run_experiment.jl planning/configs/ddpm250.yaml
#julia1.8 planning/run_experiment.jl planning/configs/ddpm500.yaml
#julia1.8 planning/run_experiment.jl planning/configs/conv1.yaml
#julia1.8 planning/run_experiment.jl planning/configs/conv8.yaml

using YAML
using HDF5
using JLD2
using Random
using MCTS
using ParticleFilters
include("minex_definition.jl")

# load the param file
config = YAML.load_file(ARGS[1])
name = config["name"] 

# Construct the POMDP
m = MinExPOMDP()

# Load the ore maps
s_all = h5read("planning/data/ore_maps.hdf5", "X")

# Load the trials states
Ntrials = config["Ntrials"]
s0_trial = [s_all[:,:,i] for i in 1:Ntrials]

# Function to load the particle set if needed
function particle_set(Nparticles)
    Random.seed!(0) # Particle set consistency
    indices = shuffle(Ntrials:size(s_all, 3))[1:Nparticles]
    return [s_all[:,:,i] for i in indices]
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
                           estimate_value=0,
                           criterion=POMCPOW.MaxUCB(config["exploration_constant"]),
                           tree_queries=config["tree_queries"],
                           k_action=config["k_action"],
                           alpha_action=config["alpha_action"],
                           k_observation=config["k_observation"],
                           alpha_observation=config["alpha_observation"]
                          )
    policy = POMDPs.solve(solver, m)
elseif config["trial_type"] == "DGM"
    include("generative_ME_belief.jl")
    initialize_DGM_python(config["DGM_path"])
    input_size=(50,50)
    up = GenerativeMEBeliefUpdater(config["model_config"], config["model_ckpt"], m, input_size)
    b0 = initialize_belief(up, nothing)
    bmdp = GenerativeBeliefMDP{typeof(m), typeof(up), typeof(b0), actiontype(m)}(m, up)
    solver = DPWSolver(next_action=MinExActionSampler(), 
                       estimate_value=0,
                       n_iterations=config["tree_queries"],
                       exploration_constant=config["exploration_constant"],
                       k_action=config["k_action"],
                       alpha_action=config["alpha_action"],
                       )
    policy = POMDPs.solve(solver, bmdp)
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
