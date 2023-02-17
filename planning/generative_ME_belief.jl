using PyCall
using Images
using POMDPs
using Random
using MCTS
include("minex_definition.jl")

function DGM_value_est(mdp, s, depth)
    if isterminal(mdp, s)
        return 0.0
    end
    Nsamples = 10
    samples = rand(s, Nsamples)
    return max(0, mean([extraction_reward(mdp.pomdp, s) for s in samples]))
end

function initialize_DGM_python(path)
    py"""
    import sys
    import torch
    sys.path.insert(0, $path)
    from src.utils import read_yaml_config_file
    from src.main_utils import instantiate_lit_model
    """
end

# Note: 1000 Seems to be the point at which paralleization breaks down on A100 GPUs
mutable struct GenerativeMEBelief
    model
    pomdp
    terminal
    drill_observations::Dict{Tuple{Int,Int},Float64}
    input_size # TODO: Remove
    function GenerativeMEBelief(config_fn, checkpoint, pomdp, input_size)
        config = py"read_yaml_config_file"(config_fn)
        # model = py"instantiate_lit_model"(config)
        py"""
        lit_model = instantiate_lit_model($config)
        try:
            lit_model = lit_model.load_from_checkpoint($checkpoint).to(torch.device("cuda"))
        except:
            lit_model = lit_model.load_from_checkpoint($checkpoint, **lit_model.hparams).to(torch.device("cuda"))
        """
        model = py"lit_model"
        # model = model.load_from_checkpoint(checkpoint).to(py"torch".device("cuda"))
        return new(model, pomdp, false, Dict{Tuple{Int,Int},Float64}(),input_size)
    end
end

function undrilled_locations(m::MinExPOMDP, b::GenerativeMEBelief)
    setdiff(m.drill_locations, keys(b.drill_observations))
end

# Used to display the state in the tree
function MCTS.node_tag(s::GenerativeMEBelief)
    return "drills = ($(s.drill_observations))"
end

struct GenerativeMEBeliefUpdater <: POMDPs.Updater 
    config_fn
    checkpoint_fn
    pomdp
    input_size #TODO: Delete
end

function POMDPs.initialize_belief(up::GenerativeMEBeliefUpdater, d)
    return GenerativeMEBelief(up.config_fn, up.checkpoint_fn, up.pomdp, up.input_size) #TODO: Delete input size
end

POMDPs.isterminal(bmdp::GenerativeBeliefMDP, b::GenerativeMEBelief) = b.terminal

# Handle vector of actions and observations in update
function POMDPs.update(up::GenerativeMEBeliefUpdater, b, a::Vector{Tuple{Int64, Int64}}, o::Vector{Float64})
    bp = deepcopy(b)
    for (a_i, o_i) in zip(a, o)
        bp.drill_observations[a_i] = o_i
    end
    return bp
end

function POMDPs.update(up::GenerativeMEBeliefUpdater, b, a, o)
    bp = deepcopy(b)
    
    if a in up.pomdp.terminal_actions 
        bp.terminal = true
    else
        bp.drill_observations[a] = o
    end
    return bp
end

# function to split indices into chunks of 1000
function split_indices(indices, batch_max=1000)
    N = length(indices)
    Nchunks = ceil(Int, N / batch_max)
    return [indices[1+(i-1)*batch_max:min(i*batch_max, N)] for i=1:Nchunks]
end

function unregulated_sample_from_model(model, drill_observations::Array)
    N = length(drill_observations)
    input = zeros(Float32, N, 2, 32, 32)
    for i=1:N
        for (a, o) in drill_observations[i]
            input[i, 1, a...] = 1
            input[i, 2, a...] = o
        end
    end
    input = py"torch".tensor(input).cuda()
    samples = model.inference_model(input).cpu().numpy()
    samples = permutedims(samples[:, 1, :, :], (2,3,1))
    return [MinExState(samples[:,:,i], collect(keys(drill_observations[i]))) for i=1:N]
end

function sample_from_model(model, drill_observations::Array, batch_max=1000)
    indices_list = split_indices(1:length(drill_observations), batch_max)
    all_samples = []
    for indices in indices_list
        println("generating ", indices, " out of ", length(drill_observations))
        push!(all_samples, unregulated_sample_from_model(model, collect(drill_observations[indices])))
    end
    return vcat(all_samples...)
end

function Base.rand(rng::AbstractRNG, b::GenerativeMEBelief, N::Int=1)
    @assert !b.terminal
    samples = sample_from_model(b.model, fill(b.drill_observations, N))
    return length(samples) == 1 ? samples[1] : samples
end
Base.rand(b::GenerativeMEBelief, N::Int=1) = rand(Random.GLOBAL_RNG, b, N)

# ## Some tests
# initialize_DGM_python("/home/acorso/Workspace/DeepGenerativeModelsCCS")
# config = "planning/models/ddpm_ore_maps_250.yaml"
# checkpoint = "planning/models/ddpm250.ckpt"
# up = GenerativeMEBeliefUpdater(config, checkpoint, MinExPOMDP(), (32,32))
# b0 = initialize_belief(up, nothing)

# # Test belief updater
# b1 = update(up, b0, (10,10), 0.9)
# @assert b1.drill_observations[(10,10)] == 0.9

# # Test belief sampling
# s1s = rand(b1, 10)
# @assert s1s[1].drill_locations == [(1,1)]

# # Test the regulated sampler
# rand(b1, 2000)

# # Test the combined sampler
# samps = unregulated_sample_from_model(b0.model, [b0.drill_observations, b1.drill_observations])

# using Plots 
# heatmap(samps[1].ore)
# heatmap(samps[2].ore)