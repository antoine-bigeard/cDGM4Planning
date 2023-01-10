using POMDPs

struct VOIPolicy <: Policy
    m
    up
    Nsamples
    VOIPolicy(m, up, Nsamples=10) = new(m, up, Nsamples)
end

Base.rand(b::ParticleCollection, N::Int) = [rand(b) for _=1:N]

function POMDPs.action(pol::VOIPolicy, b)
    vals = [] # Store the value for each action

    # Sample states to get distribution of rewards and observations
    ss = rand(b, pol.Nsamples) 

    # Loop through each action
    as = actions(pol.m)
    for a in as
        if b isa GenerativeMEBelief && a in keys(b.drill_observations) # If the action is a repeat
            push!(vals, -10)
        elseif a == :abandon
            push!(vals, 0)
        elseif a == :mine
            rs = [@gen(:r)(pol.m, s, a) for s in ss]
            push!(vals, mean(rs))
        else
            os = [@gen(:o)(pol.m, s, a) for s in ss]
            rs = [@gen(:r)(pol.m, s, a) for s in ss]

            # For each observation sample rewards from the posterior
            o_vals = []
            for o in os
                bp = update(pol.up, b, a, o)
                sps = rand(bp, pol.Nsamples)
                rps = [@gen(:r)(pol.m, sp, :mine) for sp in sps]
                push!(o_vals, max(0, mean(rps)))
            end
            push!(vals, mean(rs .+ o_vals))
        end
        println("action: ", a, " val: ", vals[end])
    end
    return as[argmax(vals)]
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
# up = GenerativeMEBeliefUpdater("planning/models/ddpm_ore_maps_250.yaml", "planning/models/ddpm250.ckpt", m, input_size)
# b0 = initialize_belief(up, nothing)

# pol = VOIPolicy(m, up)
# action(pol, b0)
