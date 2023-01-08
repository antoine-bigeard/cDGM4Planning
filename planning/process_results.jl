using JLD2
using MCTS
using ParticleFilters
using Plots
include("minex_definition.jl")
include("generative_ME_belief.jl")

# Sample POMDP with the same extraction cost and ore_threshold as the one used to generate the data
m = MinExPOMDP()

# List of results files to process
result_files = [
    "planning/results/results_Random.jld2",
    "planning/results/results_POMCPOW.jld2",
    "planning/results/results_CONV1.jld2",
    "planning/results/results_CONV8.jld2",
    "planning/results/results_DDPM250.jld2",
    "planning/results/results_DDPM500.jld2",
]

plots=[]
for rfile in result_files
    name = split(split(rfile, "_")[2], ".")[1]
    results = JLD2.load(rfile)["results"]

    # Distribution of returns
    rets = [undiscounted_reward(h) for h in results]
    p_ret = histogram(rets, bins=-150:10:200, xlabel="return", label="", title=name, xlims=(-150,200))
    vline!([mean(rets)], label="mean=$(mean(rets))", linewidth=3)

    # Regret
    results[1][1]
    extraction_rewards = [extraction_reward(m, h[1].s) for h in results]
    optimal_returns = max.(0, extraction_rewards)
    regret = optimal_returns .- rets 
    p_reg = histogram(regret, xlabel="Regret", label="", title="$name Regret", xlims=(0,200))
    vline!([mean(regret)], label="mean=$(mean(regret))", linewidth=3)

    push!(plots, plot(p_ret, p_reg, layout=(1,2)))
end

plot(plots..., layout=(length(result_files), 1), size=(1200, 400*length(result_files)))
