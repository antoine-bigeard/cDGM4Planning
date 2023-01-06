using POMCPOW

struct MinExActionSampler end

# This function is used by POMCPOW to sample a new action for DPW
# In this case, we just want to make sure that we try :mine and :abandon first before drilling
function POMCPOW.next_action(o::MinExActionSampler, problem, b, h)
    # Get the set of children from the current node
    tried_idxs = h.tree isa POMCPOWTree ? h.tree.tried[h.node] : h.tree.children[h.index]
    
    if length(tried_idxs) == 0 # First visit, try abandon
        return :abandon
    elseif length(tried_idxs) == 1 # Second visit, try mine
        return :mine
    else # 3+ visit, try drilling
        return rand(problem.drill_locations)
    end
end