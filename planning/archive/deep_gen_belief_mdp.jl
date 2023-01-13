using Plots
using POMDPs
using POMDPTools
using POMCPOW
using MCTS
using MineralExploration
using Random
using JLD2
include("generative_ME_belief.jl")

grid_dims = (50,50,1)
mainbody = BlobNode(grid_dims=grid_dims, center=MineralExploration.center_distribution(grid_dims, bounds=[grid_dims[1]/4, 3grid_dims[1]/4]))

m = MineralExplorationPOMDP(max_bores=20, delta=1, grid_spacing=0, true_mainbody_gen=mainbody, mainbody_gen=mainbody, 
                            original_max_movement=20, min_bores=2, grid_dim=grid_dims, high_fidelity_dim=grid_dims,c_exp=2)


MCTS_iterations = 1000
exploration_coefficient = 100.0
k_action = 2.0
alpha_action = 0.25

# Setup the generative belief updater
DGM = "/home/acorso/Workspace/DeepGenerativeModelsCCS"
config = "models/ddpm_ore_maps_250.yaml"
checkpoint = "models/ddpm250.ckpt"
up = GenerativeMEBeliefUpdater(DGM, config, checkpoint, m)
b0 = initialize_belief(up, nothing)
bmdp = GenerativeBeliefMDP{typeof(m), typeof(up), typeof(b0), actiontype(m)}(m, up)


# Setup the particle filter updater
ds0 = POMDPs.initialstate_distribution(m)
# up = MEBeliefUpdater(m, 1000, 2.0)
# b0 = POMDPs.initialize_belief(up, ds0)
# bmdp = GenerativeBeliefMDP(m, up)

solver = DPWSolver(n_iterations=MCTS_iterations,
                    check_repeat_action=true,
                    exploration_constant=exploration_coefficient,
                    k_action=k_action,
                    alpha_action=alpha_action,
                    tree_in_info=true,
                    estimate_value=(bmdp, b, d) -> 0,
                    show_progress=true)
planner = solve(solver, bmdp)

histories = []
for i=1:10
    s0 = rand(ds0)
    b0 = initialize_belief(up, nothing)
    results = simulate(HistoryRecorder(), m, planner, up, b0, s0)
    JLD2.@save "results_$i.jld2" results
    push!(histories, results)
end


## Plot results
function plot_map(map, actions, observations; ascale=(x)->x)
    p=heatmap(map[:,:,1]', clims=(0,1), cbar=false)
    for (a, o) in zip(actions, observations)
        scatter!([ascale(a.coords.I[1])], [ascale(a.coords.I[2])], marker_z=o.ore_quality, markerstrokecolor=:green, markersize=5, label="")
    end
    p
end

results = JLD2.load("results_2.jld2")["results"]

actions = collect(action_hist(results))
observations = collect(observation_hist(results))
rewards = collect(reward_hist(results))
states = collect(state_hist(results))

extraction = MineralExploration.extraction_reward(m, states[1])

as = []
os = []
plots = []
b = b0 
for i=0:length(actions)-2
    if i>0
        push!(as, actions[i])
        push!(os, observations[i])
        b = update(up, b, actions[i], observations[i])
    end

    gt = plot_map(states[1].ore_map, as, os)

    samps = rand(Random.GLOBAL_RNG, b, Nsamples=100)

    p1 = plot_map(samps[1].ore_map, as, os)
    p2 = plot_map(samps[2].ore_map, as, os)
    p3 = plot_map(samps[3].ore_map, as, os)
    p4 = plot_map(samps[4].ore_map, as, os)


    rs = [extraction_reward(m, s) for s in samps]
    p5 = histogram(rs, xlabel="reward", label="")
    push!(plots, gt, p1, p2, p3, p4, p5)
end
# end

actions

plot(plots..., layout=(9, 6), size=(1400, 1800))
savefig("planning_example2.png")