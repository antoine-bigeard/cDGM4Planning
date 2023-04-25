using SpillpointPOMDP
using Random
using POMDPs
using JSON
using OrderedCollections

random_policy(b, i, observations, s) = begin
	if length(observations)>0 && sum(observations[end][1:2]) > 0
		return (:stop, 0.0)
	else
		while true
			a = rand(actions(pomdp, s))
			a[1] != :stop && return a
		end
	end
end

function sim(pomdp, policy_fn, s0)
	s = deepcopy(s0)

	actions = Matrix{Float64}[zeros(length(s.m.x), 3)]
	states = [POMDPs.convert_s(Vector{Float64}, s, pomdp)]
	observations = Matrix{Float64}[zeros(length(s.m.x), 5)]
    rewards = Float64[0]

	os = []
	
	i=0
	while !isterminal(pomdp, s)
		# Produce the action
		a = policy_fn(nothing, nothing, os, s)

		# Gen the next step
		s, o, r = gen(pomdp, s, a)
		push!(os, o)

		# Store the other results
		svec = convert_s(Array{Float64}, s, pomdp)
		avec = convert_a(Array{Float64}, s, a, pomdp)
		ovec = convert_o(Array{Float64}, s, a, o, pomdp)
		push!(states, svec)
		push!(actions, avec)
		push!(observations, ovec)
		push!(rewards, r)
		i=i+1
	end
	return states, actions, observations, rewards
end

s0_dist = SubsurfaceDistribution(x = collect(range(0, 1, length=32)))

drill_locations = collect(range(0, 1, length=32))[[2:3:31...]]
obs_locs1 = collect(range(0, 1, length=32))[[8, 16, 24]]
obs_locs2 = collect(range(0, 1, length=32))[[4, 8, 12, 16, 20, 24, 28]]
pomdp = SpillpointInjectionPOMDP(;s0_dist, drill_locations, obs_configurations = [obs_locs1, obs_locs2])

pomdp.obs_configurations

Random.seed!(0) # 0 for train, 1 for val
all_results = []

all_results = Dict("states"=>[], "actions"=>[], "observations"=>[], "rewards"=>[])
for i=1:10000
	if i%1000 == 0
		println(i)
	end
	s0 = rand(initialstate(pomdp))
	ss, as, os, rs = sim(pomdp, random_policy, s0)
	println("i: ", i, " len: ", length(ss))
	push!(all_results["states"], ss)
	push!(all_results["actions"], as)
	push!(all_results["observations"], os)
	push!(all_results["rewards"], rs)
end

io = open("spillpoint_train.json", "w")
JSON.print(io, all_results)
close(io)


Random.seed!(1) # 0 for train, 1 for val
all_results = []

all_results = Dict("states"=>[], "actions"=>[], "observations"=>[], "rewards"=>[])
for i=1:100
	if i%1000 == 0
		println(i)
	end
	s0 = rand(initialstate(pomdp))
	ss, as, os, rs = sim(pomdp, random_policy, s0)
	println("i: ", i, " len: ", length(ss))
	push!(all_results["states"], ss)
	push!(all_results["actions"], as)
	push!(all_results["observations"], os)
	push!(all_results["rewards"], rs)
end

io = open("spillpoint_val.json", "w")
JSON.print(io, all_results)
close(io)
