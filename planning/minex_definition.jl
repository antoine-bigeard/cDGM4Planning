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

function POMDPs.gen(m::MinExPOMDP, s, a, rng)
    # Compute the next state
    sp = (a in m.terminal_actions || isterminal(m, s)) ? :terminal : s

    # Compute the reward
    if a == :abandon || isterminal(m, s)
        r = 0
    elseif a == :mine
        r = sum(s .> m.ore_threshold) - m.extraction_cost
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

