
using POMDPs
using Random
using MCTS

using AbstractGPs
using Random
using HDF5
using Plots
using LinearAlgebra

using StatsFuns
using Optim

using Distributions

using JLD2

using Printf

include("minex_definition.jl")

function GP_value_est(mdp, s, depth; Nsamples=500)
    if isterminal(mdp, s)
        return 0.0
    end
    samples = rand(s, Nsamples)
    extraction_rewards = [extraction_reward(mdp.pomdp, s) for s in samples]
    mean_extraction_reward = mean(extraction_rewards)
    std_extraction_reward = std(extraction_rewards)
    return max(0, mean_extraction_reward + std_extraction_reward)
end

mutable struct GenerativeGPBelief
    model
    pomdp
    terminal
    drill_observations::Dict{Tuple{Int,Int},Float64}
    init_drill_threshold::Int
    input_size # TODO: Remove
    function GenerativeGPBelief(pomdp, input_size, fname, init_drill_threshold)
        @load fname opt_f
       return new(opt_f, pomdp, false, Dict{Tuple{Int,Int},Float64}(), init_drill_threshold, input_size)
    end
end

function undrilled_locations(m::MinExPOMDP, b::GenerativeGPBelief)
    setdiff(m.drill_locations, keys(b.drill_observations))
end

function undrilled_locations(m::GenerativeBeliefMDP{P, U, B, A}, b::GenerativeGPBelief) where {P <: MinExPOMDP, U, B, A}
    undrilled_locations(m.pomdp, b)
end

# Used to display the state in the tree
function MCTS.node_tag(s::GenerativeGPBelief)
    return "#d=$(length(s.drill_observations))"
end

struct GenerativeGPBeliefUpdater <: POMDPs.Updater
    pomdp
    input_size #TODO: Delete
    fname
    init_drill_threshold
end

function POMDPs.initialize_belief(up::GenerativeGPBeliefUpdater, d)
    return GenerativeGPBelief(up.pomdp, up.input_size, up.fname, up.init_drill_threshold) #TODO: Delete input size
end

POMDPs.isterminal(b::GenerativeGPBelief) = b.terminal
POMDPs.isterminal(bmdp::GenerativeBeliefMDP, b::GenerativeGPBelief) = POMDPs.isterminal(b)

# Handle vector of actions and observations in update
function POMDPs.update(up::GenerativeGPBeliefUpdater, b, a::Vector{Tuple{Int64, Int64}}, o::Vector{Float64})
    bp = deepcopy(b)
    for (a_i, o_i) in zip(a, o)
        bp.drill_observations[a_i] = o_i
    end
    X = collect(keys(bp.drill_observations))
    X = [[i,j] for (i,j) in X]
    Y = collect(values(bp.drill_observations))

    if length(X) > 0
        fx = bp.model(X, 0.0001)
        p_fx = posterior(fx, Y)
        bp.model = p_fx
    end
    return bp
end

function POMDPs.update(up::GenerativeGPBeliefUpdater, b, a, o)
    bp = deepcopy(b)

    if a in up.pomdp.terminal_actions
        bp.terminal = true
    else
        bp.drill_observations[a] = o
    end
    X = collect(keys(bp.drill_observations))
    X = [[i,j] for (i,j) in X]
    Y = collect(values(bp.drill_observations))

    if length(X) > 0
        fx = bp.model(X, 0.0001)
        p_fx = posterior(fx, Y)
        bp.model = p_fx
    end
    return bp
end


function Base.rand(rng::AbstractRNG, b::GenerativeGPBelief, N::Int=1)
    @assert !b.terminal

    px = [[i,j] for i in 1:b.input_size[1] for j in 1:b.input_size[2]]
    μ, Σ = mean_and_cov(b.model(px))

    dist = MvNormal(μ, Σ + 1e-10*I)
    samples = []
    for _ in 1:N
        ore = reshape(rand(rng, dist), b.input_size)
        ore = transpose(ore)
        drill_locs = [(dl[1], dl[2]) for dl in keys(b.drill_observations)]
        sample = MinExState(ore, drill_locs)
        push!(samples, sample)
    end
    return length(samples) == 1 ? samples[1] : samples
end
Base.rand(b::GenerativeGPBelief, N::Int=1) = rand(Random.GLOBAL_RNG, b, N)


function POMDPTools.action_info(p::DPWPlanner, b::GenerativeGPBelief; tree_in_info=false)
    local a::actiontype(p.mdp)
    if length(b.drill_observations) < b.init_drill_threshold

        if length(b.drill_observations) == 0
            a = rand(undrilled_locations(p.mdp, b))
            info = Dict(:tree => nothing)
            return a, info
        end

        px = [[i, j] for (i,j) in undrilled_locations(p.mdp, b)]
        μ, σ = mean_and_var(b.model(px))
        μ_1σ = μ .+ σ
        ai = argmax(μ_1σ)
        a = undrilled_locations(p.mdp, b)[ai]
        info = Dict(:tree => nothing)
        return a, info
    else
        return invoke(POMDPTools.action_info, Tuple{DPWPlanner, Any}, p, b; tree_in_info=tree_in_info)
    end
end

    # # Some tests
up = GenerativeGPBeliefUpdater(MinExPOMDP(), (32,32), "gp_m52_1000.jld2", 3)
b0 = initialize_belief(up, nothing)

# Test belief updater
b1 = update(up, b0, (10,10), 0.9)
@assert b1.drill_observations[(10,10)] == 0.9

# # Test belief sampling
s1s = rand(b0)
# @assert s1s[1].drill_locations == [(10,10)]

# # # Test the regulated sampler
# rand(b1, 2000)

# using Plots
# heatmap(s1s[1].ore)
# heatmap(s1s[2].ore)
#
