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
--  3. The dropout module.
local GRUCell = function(prev_h, x, inputSize, hiddenSize, dropRate)
  -- Keep track of the linear maps we insert. This helps with parameter
  -- sharing later
  local linearMaps = {}
  local Linear = function(a, b)
    local mod = nn.Linear(a,b)
    table.insert(linearMaps, mod)
    return mod
  end

  local z_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h)
  }))
  local r_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h)
  }))

  -- For the candidate activation we need to use x mapped once more into hidden
  -- space, which is added to the mapped result of resetting prev_h.
  local h_tilda = nn.Tanh()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(nn.CMulTable()({r_gate,prev_h}))
  }))

  local dropoutMod = nn.Dropout(dropRate)
  -- The new hidden state (also the output) is a linear combination of the previous
  -- one and the updated candidate activation.
  local h = nn.CAddTable()({
    -- Cary forward some of the previous hidden state
    nn.CMulTable()({nn.AddConstant(1)(nn.MulConstant(-1)(z_gate)), prev_h}),
    -- Include some of the candidate activation
    nn.CMulTable()({z_gate, dropoutMod(h_tilda)})
  })

  return {h}, linearMaps, dropoutMod
end

return GRUCell

-- END

