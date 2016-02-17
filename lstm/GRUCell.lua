-- Make an GRU graph node.
-- 
-- For a batch size, B, inputSize, P, and hiddenSize, Q, the sizes should
-- be as follows:
--
--    x: BxP
--    prev_h: BxQ
--
-- The first two parameters are nngraph.Node instances, followed by two integers.
--
-- Returns two values:
--  1. The table containing {h} which are nngraph.Node instance. This is a table so it
--     matches LSTM semantics, which stores both h, and c.
--  2. The list of linear maps generated (used for parameter sharing).
local GRUCell = function(prev_h, x, inputSize, hiddenSize)
  -- Keep track of the linear maps we insert. This helps with parameter
  -- sharing later
  local linearMaps = {}
  local Linear = function(a, b)
    local mod = nn.Linear(a,b)
    table.insert(linearMaps, mod)
    return mod
  end

  -- Since both the update gate and reset gate involve mapping both x and prev-h, we
  -- can do these operations together.
  local xhm_squashed = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize*2)(x),
    Linear(hiddenSize,hiddenSize*2)(prev_h)
  }))

  -- Compute the update gate (z) and the reset gate (r).
  local zr_gates = nn.SplitTable(2)(nn.Reshape(2,hiddenSize)(xhm_squashed))
  local z_gate = nn.SelectTable(1)(zr_gates)
  local r_gate = nn.SelectTable(2)(zr_gates)

  -- For the candidate activation we need to use x mapped once more into hidden
  -- space, which is added to the mapped result of resetting prev_h.
  local h_tilda = nn.Tanh()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(nn.CMulTable()({r_gate,prev_h}))
  }))

  -- The new hidden state (also the output) is a linear combination of the previous
  -- one and the updated candidate activation.
  local h = nn.CAddTable()({
    -- Cary forward some of the previous hidden state
    nn.CMulTable()({nn.AddConstant(1)(nn.MulConstant(-1)(z_gate)), prev_h}),
    -- Include some of the candidate activation
    nn.CMulTable()({z_gate, h_tilda})
  })

  return {h}, linearMaps
end

return GRUCell

-- END

