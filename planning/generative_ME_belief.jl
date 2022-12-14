using PyCall
using Images

mutable struct GenerativeMEBelief
    model
    pomdp
    stopped
    decided
    rock_obs
    function GenerativeMEBelief(DGM, config, checkpoint, pomdp)
        py"""
        import sys
        import torch
        sys.path.insert(0, $DGM)
        from src.utils import read_yaml_config_file
        from src.main_utils import instantiate_lit_model
        config = read_yaml_config_file($config)
        lit_model = instantiate_lit_model(config)
        lit_model = lit_model.load_from_checkpoint($checkpoint).to(torch.device('cuda'))
        """
        return new(py"lit_model", pomdp, false, false, RockObservations())
    end
end

struct GenerativeMEBeliefUpdater <: POMDPs.Updater 
    DGM
    config 
    checkpoint
    pomdp
end

function POMDPs.initialize_belief(up::GenerativeMEBeliefUpdater, d)
    return GenerativeMEBelief(up.DGM, up.config, up.checkpoint, up.pomdp)
end

POMDPs.isterminal(bmdp::GenerativeBeliefMDP, b::GenerativeMEBelief) = b.decided

function POMDPs.update(up::GenerativeMEBeliefUpdater, b, a, o)
    bp = deepcopy(b)
    
    if a.type == :stop
        bp.stopped = true
    elseif a.type in [:mine, :abandon]
        bp.decided = true
    elseif a.type == :drill
        a_coords = reshape(Int64[a.coords[1] a.coords[2]], 2, 1)
        bp.rock_obs.coordinates = hcat(bp.rock_obs.coordinates, a_coords)
        push!(bp.rock_obs.ore_quals, o.ore_quality)
        bp.stopped = length(bp.rock_obs) >= up.pomdp.max_bores
    else
        error("unknown action type, $a")
    end
    return bp
end

function tocoords(a)
    return ceil.(Int, a .* 32 ./ 50)
end

function Base.rand(rng::AbstractRNG, b::GenerativeMEBelief; Nsamples=1)
    input = zeros(Nsamples, 2, 32, 32)
    for i in 1:length(b.rock_obs)
        cs = tocoords(b.rock_obs.coordinates[:, i])
        input[:, 1, cs...] .= 1
        input[:, 2, cs...] .= b.rock_obs.ore_quals[i]
    end
    input = py"torch.tensor($(input)).cuda()"
    samples = b.model.inference(input).cpu().numpy()
    states = []
    for i in 1:Nsamples
        ore_map = Float64.(imresize(samples[i,1,:,:], b.pomdp.grid_dim[1:2]))
        ore_map = reshape(ore_map, size(ore_map)...,1)
        push!(states, MEState(ore_map, nothing, Float64[], b.rock_obs, b.stopped, b.decided))
    end
    Nsamples ==1 ? states[1] : states
end


function POMDPs.actions(m::MineralExplorationPOMDP, b::GenerativeMEBelief)
    println("============> here!!!!")
    if b.stopped
        return MEAction[MEAction(type=:mine), MEAction(type=:abandon)]
    else
        println("heading into action")
        action_set = Set(POMDPs.actions(m))
        n_initial = length(m.initial_data)
        if !isempty(b.rock_obs.ore_quals)
            n_obs = length(b.rock_obs.ore_quals) - n_initial
            if m.max_movement != 0 && n_obs > 0
                d = m.max_movement
                drill_s = b.rock_obs.coordinates[:,end]
                x = drill_s[1]
                y = drill_s[2]
                reachable_coords = CartesianIndices((x-d:x+d,y-d:y+d))
                reachable_acts = MEAction[]
                for coord in reachable_coords
                    dx = abs(x - coord[1])
                    dy = abs(y - coord[2])
                    d2 = sqrt(dx^2 + dy^2)
                    if d2 <= d
                        push!(reachable_acts, MEAction(coords=coord))
                    end
                end
                push!(reachable_acts, MEAction(type=:stop))
                reachable_acts = Set(reachable_acts)
                # reachable_acts = Set([MEAction(coords=coord) for coord in collect(reachable_coords)])
                action_set = intersect(reachable_acts, action_set)
            end
            for i=1:n_obs
                coord = b.rock_obs.coordinates[:, i + n_initial]
                x = Int64(coord[1])
                y = Int64(coord[2])
                keepout = collect(CartesianIndices((x-m.delta:x+m.delta,y-m.delta:y+m.delta)))
                keepout_acts = Set([MEAction(coords=coord) for coord in keepout])
                setdiff!(action_set, keepout_acts)
            end
            if n_obs < m.min_bores
                delete!(action_set, MEAction(type=:stop))
            end
        elseif m.min_bores > 0
            delete!(action_set, MEAction(type=:stop))
        end
        delete!(action_set, MEAction(type=:mine))
        delete!(action_set, MEAction(type=:abandon))
        return collect(action_set)
    end
    return MEAction[]
end