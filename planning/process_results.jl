using JLD2
using MCTS
using ParticleFilters
using Plots
include("minex_definition.jl")
include("generative_ME_belief.jl")

# Sample POMDP with the same extraction cost and ore_threshold as the one used to generate the data
m = MinExPOMDP()

# List of results files to process
result_files = readdir("planning/results"; join=true)

plots=[]
for rfile in result_files
    name = split(split(rfile, "results_")[2], ".")[1]
    println("name: ", name)
    results = JLD2.load(rfile)["results"]

    # Distribution of returns
    rets = [undiscounted_reward(h) for h in results]
    p_ret = histogram(rets, bins=-150:10:200, xlabel="return", label="", title=name, xlims=(-150,200))
    vline!([mean(rets)], label="mean=$(mean(rets))", linewidth=3)

    # Regret
    extraction_rewards = [extraction_reward(m, h[1].s) for h in results]
    optimal_returns = max.(0, extraction_rewards)
    regret = optimal_returns .- rets 
    p_reg = histogram(regret, xlabel="Regret", label="", title="$name Regret", xlims=(0,200))
    vline!([mean(regret)], label="mean=$(mean(regret))", linewidth=3)

    # Number of actions taken
    len_actions = [length(action_hist(h)) for h in results]
    p_acts = histogram(len_actions, xlabel="No. Actions", label="", title=name)
    
    push!(plots, plot(p_ret, p_reg, p_acts, layout=(1,3)))
end

plot(plots..., layout=(length(result_files), 1), size=(1800, 400*length(result_files)))


## Checkout trees
# result_file = "planning/results/results_POMCPOW_Loose.jld2"
# result = JLD2.load(result_file)["results"]
# trees = [D3Tree(h.action_info[:tree]) for h in result[1]]
# inchrome(trees[2])