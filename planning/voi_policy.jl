using POMDPs
using ParticleFilters
include("generative_ME_belief.jl")

function mc_multi_actions(pomdp, b, N_mc_actions, N_max_drills=10)
    drill_locations = undrilled_locations(pomdp, b)
    drill_actions = []
    for _=1:N_mc_actions
        Ndrills = min(N_max_drills, length(drill_locations))
        if Ndrills >= 1
            a = sample(drill_locations, rand(1:Ndrills), replace=false)
            push!(drill_actions, a)
        end
    end
    unique!(drill_actions)
    return [:abandon, :mine, drill_actions...]
end

struct VOIPolicy <: Policy
    m
    up
    Nobs
    Nsamples_est
    action_fn
    VOIPolicy(m, up, Nobs=10, Nsamples_est=10, action_fn=(b)->actions(m, b)) = new(m, up, Nobs, Nsamples_est, action_fn)
end
VOIMultiActionPolicy(m, up, Nobs=10, Nsamples_est=10, N_mc_actions=100, action_fn=(b)->mc_multi_actions(m, b, N_mc_actions)) = VOIPolicy(m, up, Nobs, Nsamples_est, action_fn)

Base.rand(b::ParticleCollection, N::Int) = [rand(b) for _=1:N]

# Default Implementation for the particle filter and other belief types
# beliefs_per_action is an array where the index corresponds to the action and the elements are vectors of beliefs
# Nsamples is the number of samples to draw from each and every belief
function gen_belief_samples(beliefs_per_action, Nsamples)
    # We will construct a 3x nested array where the top index is action, followed by belief, followed by sample
    sps_per_belief_per_action = []

    # We start by looping over actions and pulling out the array of beliefs for that action
    for beliefs in beliefs_per_action
        # We will construct a 2x nested array where the top index is belief, followed by sample
        sps_per_belief = []
        for b in beliefs
            # Sample Nsamples from this belief (creates an array of samples)
            sps = rand(b, Nsamples)
            # Push that array of samples back to our 2x nested array
            push!(sps_per_belief, sps)
        end
        # Push the 2x nested array back to our 3x nested array
        push!(sps_per_belief_per_action, sps_per_belief)
    end
    return sps_per_belief_per_action
end

# This implementation is optimized for the deep generative model
function gen_belief_samples(beliefs_per_action::Vector{Vector{GenerativeMEBelief}}, Nsamples)
    # Flatten the set of beliefs
    beliefs = vcat(beliefs_per_action...)

    # pull out their observations
    drill_obs = [b.drill_observations for b in beliefs]
    
    # repeat the obs for the number of samples we want
    drill_obs = repeat(drill_obs, inner=Nsamples)

    # Generate the samples
    sps = sample_from_model(beliefs[1].model, drill_obs)

    # Reshape the samples into the 3x nested array
    spindex=1
    sps_per_belief_per_action = []
    for ai in 1:length(beliefs_per_action)
        sps_per_belief = []
        for bi in beliefs_per_action[ai]
            push!(sps_per_belief, sps[spindex:spindex+Nsamples-1])
            spindex += Nsamples
        end
        push!(sps_per_belief_per_action, sps_per_belief)
    end

    return sps_per_belief_per_action
end

function POMDPs.action(pol::VOIPolicy, b)
    ss = rand(b, pol.Nobs) 

    # Get the value of abandoning and mining
    abandon_value = 0.0
    mine_value = mean([@gen(:r)(pol.m, s, :mine) for s in ss])

    # Get the available drilling locationsactions
    drill_actions = setdiff(pol.action_fn(b), [:abandon, :mine])
    
    beliefs_per_action = []
    rewards_per_action = []
    for a in drill_actions
        println("actions: ", a, " out of, ", length(drill_actions))
        # Generate observations and rewards from our set of states
        samps = [@gen(:o, :r)(pol.m, s, a) for s in ss]

        # For each observation, do an update and store the belief and reward
        bps = [update(pol.up, b, a, o) for (o,_) in samps]
        rs = [r for (_,r) in samps]
        push!(beliefs_per_action, bps)
        push!(rewards_per_action,  rs)
    end

    # Process the samples to compute the value of each action
    println("generating samples from beliefs...")
    beliefs_per_action = [beliefs_per_action...]
    sps_per_belief_per_action = gen_belief_samples(beliefs_per_action, pol.Nsamples_est)
    values_per_action = []
    for ai=1:length(drill_actions)
        sps_per_belief = sps_per_belief_per_action[ai]
        rs = rewards_per_action[ai]
        belief_vals = []
        for sps in sps_per_belief
            rps = [@gen(:r)(pol.m, sp, :mine) for sp in sps]
            push!(belief_vals, max(0, mean(rps)))
        end
        push!(values_per_action, mean(rs .+ belief_vals))
    end

    # Combine the values and terminal actions and choose the highest value one
    all_vals  = [abandon_value, mine_value, values_per_action...]
    all_acts = [:abandon, :mine, drill_actions...]
    return all_acts[argmax(all_vals)]
end



## Code to run a few simple tests
# m = MinExPOMDP()

# # Particle Filter VoI
# Nparticles = 10000
# b0 = ParticleCollection(particle_set(Nparticles))
# up = BootstrapFilter(m, Nparticles)

# pol = VOIPolicy(m, up)
# action(pol, b0)


# # DGM VoI
# up = GenerativeMEBeliefUpdater("planning/models/ddpm_ore_maps_250.yaml", "planning/models/ddpm250.ckpt", m, (32,32))
# b0 = initialize_belief(up, nothing)

# pol = VOIPolicy(m, up)
# action(pol, b0)
