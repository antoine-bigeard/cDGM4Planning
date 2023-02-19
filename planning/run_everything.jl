using HDF5
using JLD2
using Random
using MCTS
using ParticleFilters
using Images
using Random
include("minex_definition.jl")
include("voi_policy.jl")
include("generative_ME_belief.jl")
initialize_DGM_python("/home/acorso/Workspace/cDGM4Planning/")

# Params
tree_queries = 1000
exploration_constant = 100.0
k_action = 4.0
alpha_action = 0.5
k_state = 2.0
alpha_state = 0.25
k_observation = 2.0
alpha_observation = 0.25
Nobs_VOI = 50
Nsamples_est_VOI = 10
N_mc_actions_VOI = 50
Ntest = 50
savefolder="results"

# Function for running the experiments and saving what we want to
function runsim(m, policy, up, b0, s0)
    hist = Dict(:b0=>b0, :s0 => s0, :as =>[], :os=>[], :rs=>[])
    s = deepcopy(s0)
    b = deepcopy(b0)
    t = @elapsed while !isterminal(m, s)
        a = action(policy, b)
        s, o, r = gen(m, s, a, Random.GLOBAL_RNG)
        b = update(up, b, a, o)

        hist[:as] = [hist[:as]; a]
        hist[:os] = [hist[:os]; o]
        hist[:rs] = [hist[:rs]; r]
    end
    hist[:time] = t
    return hist
end

# Function to generate DGM belief
function gen_DGM_belief(config, checkpoint)
    m = MinExPOMDP()
    up = GenerativeMEBeliefUpdater(config, checkpoint, m, (32,32)) #TODO: remove input size
    b0 = initialize_belief(up, nothing)
    m, up, b0
end

# Load the ore maps
train_ore_maps = imresize(h5read("planning/data/ore_maps.hdf5", "X"), (32,32))
test_ore_maps = imresize(h5read("planning/data/test_ore_maps.hdf5", "X"), (32,32))

# Load the test states
s0_test = [MinExState(test_ore_maps[:,:,i]) for i in 1:Ntest]

# Load all of the particle filter representations
particle_counts = [1_000, 10_000, 100_000]
abc_params = [0.1, 0.05, 0.01]

# Load all of the experiments
experiments = []

# Fill random experiment
function rand_pol()
    m = MinExPOMDP()
    up = NothingUpdater()
    b0 = initialize_belief(NothingUpdater(), nothing)
    policy = RandomPolicy(m)
    m, up, b0, policy
end
push!(experiments, ["Random", rand_pol])

# Fill pomcpow trials
for pc in particle_counts, abc in abc_params
    function pompow_pf()
        m = MinExPOMDP(σ_abc=abc)
        b = ParticleCollection([MinExState(train_ore_maps[:,:,i]) for i in 1:pc])
        up = BootstrapFilter(m, pc)
        solver = POMCPOWSolver(next_action=MinExActionSampler(), 
                           estimate_value=(pomdp, s, h, steps) -> isterminal(pomdp, s) ? 0 : max(0, extraction_reward(pomdp, s)),
                           criterion=POMCPOW.MaxUCB(exploration_constant),
                           tree_queries=tree_queries,
                           k_action=k_action,
                           alpha_action=alpha_action,
                           k_observation=k_observation,
                           alpha_observation=alpha_observation,
                           tree_in_info=false,
                          )
        policy = POMDPs.solve(solver, m)
        m, up, b, policy
    end
    push!(experiments, ["POMCPOW-Particle-$(pc)-$abc", pompow_pf])
end

# Fill pf voi multi experiments
for pc in particle_counts, abc in abc_params
    function voi_pf()
        m = MinExPOMDP(σ_abc=abc)
        b = ParticleCollection([MinExState(train_ore_maps[:,:,i]) for i in 1:pc])
        up = BootstrapFilter(m, pc)
        policy = VOIMultiActionPolicy(m, up, Nobs_VOI, Nsamples_est_VOI, N_mc_actions_VOI)
        m, up, b, policy
    end
    push!(experiments, ["VOI-Particle-$(pc)-$abc", voi_pf])
end

# Fill in DGM voi multi experiments
for name in reverse(readdir("planning/models"; join=false))
    function voi_dgm() 
        m, up, b = gen_DGM_belief("planning/models/$name/$name.yaml", "planning/models/$name/$name.ckpt")
        policy = VOIMultiActionPolicy(m, up, Nobs_VOI, Nsamples_est_VOI, N_mc_actions_VOI)
        m, up, b, policy
    end
    push!(experiments, ["VOI-$name", voi_dgm])
end

# Fill DGM tree search experiments
for name in reverse(readdir("planning/models"; join=false))
    function treesearch_dgm() 
        m, up, b = gen_DGM_belief("planning/models/$name/$name.yaml", "planning/models/$name/$name.ckpt")
        bmdp = GenerativeBeliefMDP{typeof(m), typeof(up), typeof(b), actiontype(m)}(m, up)
        solver = DPWSolver(next_action=MinExActionSampler(), 
                       estimate_value=DGM_value_est,
                       n_iterations=tree_queries,
                       exploration_constant=exploration_constant,
                       k_action=k_action,
                       alpha_action=alpha_action,
                       k_state=k_state,
                       alpha_state=alpha_state,
                       tree_in_info=true,
                       )
    policy = POMDPs.solve(solver, bmdp)
    m, up, b, policy
    end
    push!(experiments, ["treesearch-$name", treesearch_dgm])
end

# Run the trials
for (name, experiment) in experiments
    # Skip trials that have completed
    if isfile("$savefolder/results_$name.jld2")
        println("Skipping $name...")
        continue
    end

    println("Running $name...")
    m, up, b0, policy = experiment()
    results = []
    for (i,s0) in enumerate(s0_test)
        println("Running trial $i of $name...")
        push!(results, runsim(m, policy, up, b0, s0))

        # Save all results after every iteration
        println("Saving trial $i of $name...")
        JLD2.save("$savefolder/partialresults_$name.jld2", Dict("results" => results)) 
    end
    JLD2.save("$savefolder/results_$name.jld2", Dict("results" => results)) 
    rm("$savefolder/partialresults_$name.jld2")
end
