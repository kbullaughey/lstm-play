-- Make an LSTM graph node.
-- 
-- The MemoryCell can either work in batch mode (timeAxis=2) or single sequence
-- mode (timeAxis=1).
--
-- In batch mode with batchSize, B, inputSize, F, and hiddenSize, H, the sizes should
-- be as follows:
--
--    x: BxF
--    prev_h: BxH
--    prev_c: BxH
--
-- In non-batch mode, we have:
--
--    x: vector length F
--    h: vector length H
--    c: vector length H
--
-- The first three parameters are nngraph.Node instances, followed by two
-- integers and a boolean.
--
-- Returns two values:
--  1. The table containing {h, c} which are nngraph.Node instance.
--  2. The list of linear maps generated (used for parameter sharing).
local MemoryCell = function(prev_h, prev_c, x, inputSize, hiddenSize)
  -- Keep track of the linear maps we insert. This helps with parameter
  -- sharing later
  local linearMaps = {}
  local Linear = function(a, b)
    local mod = nn.Linear(a,b)
    table.insert(linearMaps, mod)
    return mod
  end

  -- Equation numbers refer to Graves (2014) Generating sequences with recurrent
  -- neural networks.

  -- Since x and prev_h are each fed through four linear maps, we can do the four maps
  -- together and later split them out. Also, since x mapped to hidden-space and
  -- prev_h mapped to hidden-space are always added, we can do that up-front too.
  local xhm = nn.CAddTable()({
    Linear(inputSize,hiddenSize*4)(x),
    Linear(hiddenSize,hiddenSize*4)(prev_h)
  })

  -- Split the four xh mappings appart.
  -- In batch mode, the reshaped tensor will be Bx4xH, in non-batch mode, 4xH.
  local xhm_split = nn.SplitTable(-2)(nn.Reshape(4,hiddenSize)(xhm))

  -- Input gate. Equation (7)
  local i_gate = nn.Sigmoid()(nn.CAddTable()({
    nn.SelectTable(1)(xhm_split),
    Linear(hiddenSize,hiddenSize)(prev_c)
  }))

  -- Forget gate. Equation (8)
  local f_gate = nn.Sigmoid()(nn.CAddTable()({
    nn.SelectTable(2)(xhm_split),
    Linear(hiddenSize,hiddenSize)(prev_c)
  }))

  -- New contribution to c. Right term in equation (9)
  local learning = nn.Tanh()(nn.SelectTable(3)(xhm_split))

  -- Memory cell. Equation (9)
  local c = nn.CAddTable()({
    nn.CMulTable()({f_gate, prev_c}),
    nn.CMulTable()({i_gate, learning})
  })

  -- Output gate. Equation (10)
  local o_gate = nn.Sigmoid()(nn.CAddTable()({
    nn.SelectTable(4)(xhm_split),
    Linear(hiddenSize,hiddenSize)(c)
  }))

  local h = nn.CMulTable()({o_gate, nn.Tanh()(c)})

  return {h=h, c=c}, linearMaps
end

return MemoryCell

-- END
