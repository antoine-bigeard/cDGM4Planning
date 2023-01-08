using POMDPs
using POMDPTools
using Distributions
using Parameters

## Definition of the POMDP
@with_kw struct MinExPOMDP <: POMDP{Any, Any, Any} 
    ore_threshold = 0.7
    extraction_cost = 150
    drill_cost = 0.1
    drill_locations = [(i,j) for i=5:5:45 for j=5:5:45]
    terminal_actions = [:abandon, :mine]
    σ_abc = 0.1
    γ=0.999
end

POMDPs.discount(m::MinExPOMDP) = m.γ
POMDPs.actions(m::MinExPOMDP) = [m.terminal_actions..., m.drill_locations...]

POMDPs.isterminal(m::MinExPOMDP, s) = s == :terminal

function extraction_reward(m, s)
    sum(s .> m.ore_threshold) - m.extraction_cost
end

function POMDPs.gen(m::MinExPOMDP, s, a, rng)
    # Compute the next state
    sp = (a in m.terminal_actions || isterminal(m, s)) ? :terminal : s

    # Compute the reward
    if a == :abandon || isterminal(m, s)
        r = 0
    elseif a == :mine
        r = extraction_reward(m, s)
    else
        r = -m.drill_cost
    end

    # observation
    if isterminal(m, s) || a in m.terminal_actions
        o=nothing
    else
        o = s[a...]
    end

    return (;sp, o, r)
end

function POMDPTools.obs_weight(m::MinExPOMDP, s, a, sp, o)
    if (isterminal(m, s) || a in m.terminal_actions)
        w = Float64(isnothing(o))
    else
        w = pdf(Normal(s[a...], m.σ_abc), o)
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
    
    if length(tried_idxs) == 0 # First visit, try abandon
        return :abandon
    elseif length(tried_idxs) == 1 # Second visit, try mine
        return :mine
    else # 3+ visit, try drilling
        if problem isa MinExPOMDP
            return rand(problem.drill_locations)
        elseif problem isa GenerativeBeliefMDP
            return rand(problem.pomdp.drill_locations)
        else
            error("Didn't recognize problem type")
        end
    end
end