using POMDPs
using POMDPTools
using Distributions
using Parameters

# Helper function for sampling multiple states from teh posterior
Base.rand(b::ParticleCollection, N::Int) = [rand(b) for _=1:N]

## Definition of the POMDP
@with_kw struct MinExPOMDP <: POMDP{Any, Any, Any} 
    ore_threshold = 0.7
    extraction_cost = 52 # for 32x32. use 150 for 50x50
    drill_cost = .1
    drill_locations = [(i,j) for i=5:5:30 for j=5:5:30]
    terminal_actions = [:abandon, :mine]
    σ_abc = 0.1
    γ=0.999
end

mutable struct MinExState
    ore
    drill_locations::Vector{Tuple{Int, Int}}
    MinExState(ore, drill_locations=[]) = new(ore, drill_locations)
end

Base.copy(s::MinExState) = MinExState(s.ore, deepcopy(s.drill_locations))

POMDPs.discount(m::MinExPOMDP) = m.γ


function undrilled_locations(m::MinExPOMDP, b)
    undrilled_locations(m::MinExPOMDP, rand(b))
end

function undrilled_locations(m::MinExPOMDP, s::MinExState)
    setdiff(m.drill_locations, s.drill_locations)
end

function POMDPs.actions(m::MinExPOMDP, s_or_b)
    [m.terminal_actions..., undrilled_locations(m, s_or_b)...]
end

function POMDPs.actions(m::MinExPOMDP)
    println("WARNING: calling POMDPs.actions without state or belief")
    [m.terminal_actions..., m.drill_locations]
end

POMDPs.isterminal(m::MinExPOMDP, s) = s == :terminal

function extraction_reward(m, s)
    sum(s.ore .> m.ore_threshold) - m.extraction_cost
end

# This gen function is for passing multiple drilling actions
function POMDPs.gen(m::MinExPOMDP, s, as::Vector{Tuple{Int, Int}}, rng)
    rtot = 0
    os = Float64[]
    for a in as
        s, o, r = gen(m, s, a, rng)
        push!(os, o)
        rtot += r
    end
    return (;sp=s, o=os, r=rtot)
end

function POMDPs.gen(m::MinExPOMDP, s, a, rng)
    # Compute the next state
    sp = (a in m.terminal_actions || isterminal(m, s)) ? :terminal : copy(s)
    
    # Compute the reward
    if a == :abandon || isterminal(m, s)
        r = 0
    elseif a == :mine
        r = extraction_reward(m, s)
    else
        push!(sp.drill_locations, a)
        r = -m.drill_cost
    end

    # observation
    if isterminal(m, s) || a in m.terminal_actions
        o=nothing
    else
        o = s.ore[a...]
    end

    return (;sp, o, r)
end

# Function for handling vector of actions (and therefore vector of observations)
function POMDPTools.obs_weight(m::MinExPOMDP, s, a::Vector{Tuple{Int64, Int64}}, sp, o::Vector{Float64})
    w = 1.0
    @elapsed for (a_i, o_i) in zip(a, o)
        w *= obs_weight(m, s, a_i, sp, o_i)
    end
    return w
end

function POMDPTools.obs_weight(m::MinExPOMDP, s, a, sp, o)
    if (isterminal(m, s) || a in m.terminal_actions)
        w = Float64(isnothing(o))
    else
        w = pdf(Normal(s.ore[a...], m.σ_abc), o)
    end
    return w
end

## Next action functionality for tree-search solvers 
using POMCPOW

struct MinExActionSampler end

# This function is used by POMCPOW to sample a new action for DPW
# In this case, we just want to make sure that we try :mine and :abandon first before drilling
function POMCPOW.next_action(o::MinExActionSampler, problem, b, h)
    # Get the set of children from the current node
    tried_idxs = h.tree isa POMCPOWTree ? h.tree.tried[h.node] : h.tree.children[h.index]
    drill_locations = undrilled_locations(problem, b)
    if length(tried_idxs) == 0 # First visit, try abandon
        return :abandon
    elseif length(drill_locations) == 0 || length(tried_idxs) == 1 # Second visit, try mine
        return :mine
    else # 3+ visit, try drilling
        return rand(drill_locations)
    end
end

## Some tests
# using Random
# m = MinExPOMDP()

# # Test the gen function with a single action
# s = MinExState(rand(32, 32))
# a = (5, 5)
# sp, o, r = gen(m, s, a, Random.GLOBAL_RNG)
# @assert sp.ore[a...] == o

# @assert actions(m, sp) == setdiff(actions(m, s), [a])

# obs_weight(m, s, a, sp, o)

# # Test the gen function with multiple actions
# s = MinExState(rand(32, 32))
# as = [(5, 5), (5, 10), (5, 15)]
# sp, os, r = gen(m, s, as, Random.GLOBAL_RNG)
# @assert length(os) == length(as)
# @assert actions(m, sp) == setdiff(actions(m, s), as)

# obs_weight(m, s, a, sp, o)
