include("generative_ME_belief.jl")
include("minex_definition.jl")

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

function show_samples(config, checkpoint)
    up = GenerativeMEBeliefUpdater(DGM, config, checkpoint, m, input_size)
    b0 = initialize_belief(up, nothing)
end


b = show_samples("models/ddpm_ore_maps_250.yaml", "models/ddpm250.ckpt")

rand(Random.GLOBAL_RNG, b)
show_samples("models/ddpm_ore_maps_500.yaml", "models/ddpm500.ckpt")
show_samples("models/config_conv.yaml", "models/halfinject_conv.ckpt")
show_samples("models/config_conv8.yaml", "models/halfinject_conv8.ckpt")


