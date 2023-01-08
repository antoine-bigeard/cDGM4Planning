using PyCall
using Images
using POMDPs
using Random

function initialize_DGM_python(path)
    py"""
    import sys
    import torch
    sys.path.insert(0, $path)
    from src.utils import read_yaml_config_file
    from src.main_utils import instantiate_lit_model
    """
end


mutable struct GenerativeMEBelief
    model
    pomdp
    terminal
    drill_observations
    input_size
    function GenerativeMEBelief(config_fn, checkpoint, pomdp, input_size)
        config = py"read_yaml_config_file"(config_fn)
        model = py"instantiate_lit_model"(config)
        model = model.load_from_checkpoint(checkpoint).to(py"torch".device("cuda"))
        return new(model, pomdp, false, Dict(),input_size)
    end
end

struct GenerativeMEBeliefUpdater <: POMDPs.Updater 
    config_fn
    checkpoint_fn
    pomdp
    input_size
end

function POMDPs.initialize_belief(up::GenerativeMEBeliefUpdater, d)
    return GenerativeMEBelief(up.config_fn, up.checkpoint_fn, up.pomdp, up.input_size)
end

POMDPs.isterminal(bmdp::GenerativeBeliefMDP, b::GenerativeMEBelief) = b.terminal

function POMDPs.update(up::GenerativeMEBeliefUpdater, b, a, o)
    bp = deepcopy(b)
    
    if a in up.pomdp.terminal_actions 
        bp.terminal = true
    else
        bp.drill_observations[a] = o
    end
    return bp
end

function tocoords(a, size)
    return ceil.(Int, a .* (32,32) ./ size)
end

function Base.rand(rng::AbstractRNG, b::GenerativeMEBelief; Nsamples=1)
    @assert !b.terminal
    input = zeros(Nsamples, 2, 32, 32)
    for (drill_loc, obs) in b.drill_observations
        loc = tocoords(drill_loc, b.input_size)
        input[:, 1, loc...] .= 1
        input[:, 2, loc...] .= obs
    end
    input = py"torch".tensor(input).cuda()
    samples = b.model.inference_model(input).cpu().numpy()
    ret = Nsamples == 1 ? samples[1, 1, :,:] : permutedims(samples[:, 1, :, :], (2,3,1))
    imresize(ret, (50,50))
end
