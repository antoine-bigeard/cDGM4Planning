using Random
using MCTS
using ParticleFilters
include("minex_definition.jl")
include("generative_ME_belief.jl")
initialize_DGM_python("/home/acorso/Workspace/DeepGenerativeModelsCCS")

# Params
input_size=(50,50)
tree_queries=1000
exploration_constant=100.0
k_action = 2.0
alpha_action = 0.25
Ntrials=100

# Construct the POMDP
m = MinExPOMDP()

# Load the ore maps
s_all = h5read("planning/data/ore_maps.hdf5", "X")

# Load the trials states
s0_trial = [s_all[:,:,i] for i in 1:Ntrials]


up = GenerativeMEBeliefUpdater("planning/models/ddpm_ore_maps_250.yaml", "planning/models/ddpm250.ckpt", m, input_size)
b0 = initialize_belief(up, nothing)
bmdp = GenerativeBeliefMDP{typeof(m), typeof(up), typeof(b0), actiontype(m)}(m, up)
solver = DPWSolver(;next_action=MinExActionSampler(), 
                    estimate_value=0,
                    n_iterations=tree_queries,
                    exploration_constant,
                    k_action,
                    alpha_action,
                    tree_in_info=true
                    )
policy = POMDPs.solve(solver, bmdp)

results = []
for i=1:5
    push!(results, simulate(HistoryRecorder(), m, deepcopy(policy), deepcopy(up), deepcopy(b0), s0))
    JLD2.save("DDPM250_w_tree.jld2", Dict("results" => results)) 
end

using D3Trees

results = JLD2.load("DDPM250_w_tree.jld2")["results"]

inchrome(D3Tree(results[1][1].action_info[:tree]))

name = "DDPM250"
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
plot(p_ret, p_reg, layout=(1,2), size=(1200,400))
savefig("trial_w_5.pdf")

