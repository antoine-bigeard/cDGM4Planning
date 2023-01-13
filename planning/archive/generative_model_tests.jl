include("generative_ME_belief.jl")
include("minex_definition.jl")
using Plots

function plot_map(map, actions, observations; ascale=(x)->x)
    p=heatmap(map[:,:,1]', clims=(0,1), cbar=false)
    for (a, o) in zip(actions, observations)
        scatter!([ascale(a.coords.I[1])], [ascale(a.coords.I[2])], marker_z=o.ore_quality, markerstrokecolor=:green, markersize=5, label="")
    end
    p
end


m = MinExPOMDP()
input_size = (50,50)

DGM = "/home/acorso/Workspace/DeepGenerativeModelsCCS"

function gen_DGM_belief(config, checkpoint, Nsamples=5)
    up = GenerativeMEBeliefUpdater(DGM, config, checkpoint, m, input_size)
    b0 = initialize_belief(up, nothing)
    up, b0
end

function plot_samples(samps)
    Nsamps = size(samps, 3)
    plot([heatmap(samps[:,:,i]', clims=(0,1), cbar=false) for i=1:Nsamps]..., layout=(Nsamps, 1))
end


ddpm250_samps = gen_samples("planning/models/ddpm_ore_maps_250.yaml", "planning/models/ddpm250.ckpt")
ddpm500_samps = gen_samples("planning/models/ddpm_ore_maps_500.yaml", "planning/models/ddpm500.ckpt")
conv1_samps = gen_samples("planning/models/config_conv.yaml", "planning/models/halfinject_conv.ckpt")
conv8_samps = gen_samples("planning/models/config_conv8.yaml", "planning/models/halfinject_conv8.ckpt")

p1 = plot_samples(ddpm250_samps)
p2 = plot_samples(ddpm500_samps)
p3 = plot_samples(conv1_samps)
p4 = plot_samples(conv8_samps)

plot(p1,p2,p3,p4, layout=(1,4), size=(400, 500))