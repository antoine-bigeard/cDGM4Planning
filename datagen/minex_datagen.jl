# Note that as of 10/29/22 this uses the branch "moss" from MineralExploration
using MineralExploration
using Plots
using Statistics
using Random
using JLD2
using POMDPs

N_INITIAL = 0
MAX_BORES = 2
MIN_BORES = 2
GRID_SPACING = 0
MAX_MOVEMENT = 20

grid_dims = (50, 50, 1)
true_mainbody = BlobNode(grid_dims=grid_dims, 
						 factor=4, 
						 center=MineralExploration.center_distribution(grid_dims, bounds=[grid_dims[1]/4, 3grid_dims[1]/4]))
mainbody = BlobNode(grid_dims=grid_dims)

m = MineralExplorationPOMDP(max_bores=MAX_BORES, delta=GRID_SPACING+1, grid_spacing=GRID_SPACING,
                            true_mainbody_gen=true_mainbody, mainbody_gen=mainbody, original_max_movement=MAX_MOVEMENT,
                            min_bores=MIN_BORES, grid_dim=grid_dims, c_exp=2)

ds0 = POMDPs.initialstate_distribution(m)


anim = @animate for i=1:100
	s0 = rand(ds0; truth=true)
	heatmap(s0.ore_map[:,:,1])
end

gif(anim)


s, r = [], []
for i=1:5000
	s0 = rand(ds0; truth=true) #Checked
	push!(s, s0.ore_map)
	push!(r, MineralExploration.extraction_reward(m, s0))
end

histogram(r)

m.drill_cost

JLD2.save("data/minex_data.jld2", Dict("s"=>s, "r"=>r))

using HDF5


Random.seed!(1234555)
train_x = zeros(Float32, 50, 50, 100)
for i=1:size(train_x, 3)
	s0 = rand(ds0; truth=true) #Checked
	train_x[:,:,i] = s0.ore_map
end

using HDF5
fid = h5open("test_ore_maps.hdf5", "w")
write(fid, "X", train_x)
close(fid)

