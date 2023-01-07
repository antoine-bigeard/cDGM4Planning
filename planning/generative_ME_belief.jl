using PyCall
using Images
using POMDPs
using Random

mutable struct GenerativeMEBelief
    model
    pomdp
    terminal
    drill_observations
    input_size
    function GenerativeMEBelief(DGM, config, checkpoint, pomdp, input_size)
        py"""
        import sys
        import torch
        torch.cuda.set_device(1)
        sys.path.insert(0, $DGM)
        from src.utils import read_yaml_config_file
        from src.main_utils import instantiate_lit_model
        config = read_yaml_config_file($config)
        lit_model = instantiate_lit_model(config)
        lit_model = lit_model.load_from_checkpoint($checkpoint).to(torch.device('cuda'))
        """
        return new(py"lit_model", pomdp, false, Dict(),input_size)
    end
end

struct GenerativeMEBeliefUpdater <: POMDPs.Updater 
    DGM
    config 
    checkpoint
    pomdp
    input_size
end

function POMDPs.initialize_belief(up::GenerativeMEBeliefUpdater, d)
    return GenerativeMEBelief(up.DGM, up.config, up.checkpoint, up.pomdp, up.input_size)
end

POMDPs.isterminal(bmdp, b::GenerativeMEBelief) = b.terminal

function POMDPs.update(up::GenerativeMEBeliefUpdater, b, a, o)
    bp = deepcopy(b)
    
    if a in up.pomdp.terminal_actions 
        bp.terminal = true
    else
        bp.drill_observations[a] = o
    end
    return bp
end

function Base.rand(rng::AbstractRNG, b::GenerativeMEBelief; Nsamples=1)
    input = zeros(Nsamples, 2, b.input_size...)
    for (drill_loc, obs) in b.drill_observations
        input[:, 1, drill_loc...] .= 1
        input[:, 2, drill_loc...] .= obs
    end
    input = py"torch.tensor($(input)).cuda()"
    samples = b.model.inference(input).cpu().numpy()
    states = []
    for i in 1:Nsamples
        ore_map = Float64.(imresize(samples[i,1,:,:], b.pomdp.grid_dim[1:2]))
        ore_map = reshape(ore_map, size(ore_map)...,1)
        push!(states, MEState(ore_map, nothing, Float64[], b.rock_obs, b.stopped, b.decided))
    end
    Nsamples == 1 ? states[1] : states
end
