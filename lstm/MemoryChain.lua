local stringx = require 'pl.stringx'

-- Constructor. It takes four parameters:
--  1. inputSize [integer] - width of input vector
--  2. hiddenSizes [table of integers] - each entry in the table gives the
--      width of the hidden state for a layer. The number of LSTM layers
--      is determined from the size of this table. The width of the
--      memory cell is the same as the hidden state size for each layer.
--  3. batchSize [integer] - The number of examples in a mini-batch. At present
--      this implementation only works with mini-batches.
--  4. maxLength [integer] - The length of the longest sequence we should
--      expect to see. This is necessary because we pre-create all the
--      memory cells and use the same ones for all sequences.
local MemoryChain = function(inputSize, hiddenSizes, batchSize, maxLength)
  print("MemoryChain(" .. inputSize .. ',<' .. stringx.join(',',hiddenSizes) ..
    '>,' .. batchSize .. "," .. maxLength .. ')')

  if #hiddenSizes ~= 1 then
    error("current implementation only works with one layer")
  end
  local hiddenSize = hiddenSizes[1]

  -- Keep a namespace.
  local ns = {inputSize=inputSize, hiddenSize=hiddenSize, maxLength=maxLength,
    batchSize=batchSize}

  -- This will be the initial h (probably set to zeros)
  ns.initial_h = nn.Identity()()
  ns.initial_c = nn.Identity()()

  -- The length indicators is a one-hot representation of where each sequence
  -- ends. It is a BxL matrix with exactly one 1 in each row.
  ns.lengthIndicators = nn.Identity()()

  -- This will be expecting a matrix of size BxLxI where B is the batch size,
  -- L is the sequence length, and I is the number of inputs.
  ns.inputs = nn.Identity()()
  ns.splitInputs = nn.SplitTable(2)(ns.inputs)

  -- We Save all the hidden states in this table for use in prediction.
  ns.hiddenStates = {}
  ns.linearMaps = {}

  -- Iterate over the anticipated sequence length, creating a unit for each
  -- timestep.
  local unit = {h=ns.initial_h, c=ns.initial_c}
  for t=1,maxLength do
    local x = nn.SelectTable(t)(ns.splitInputs)
    unit, ns.linearMaps[t] = lstm.MemoryCell(unit.h, unit.c, x, inputSize, hiddenSize)

    -- Add a middle dimension to prepare the hidden states it to get joined
    -- along the time steps as the middle dimension.
    ns.hiddenStates[t] = nn.Reshape(batchSize, 1, hiddenSize)(unit.h)
  end

  -- Paste all the hidden matricies together. Each one is BxH and the result
  -- will be BxLxH
  ns.out = nn.JoinTable(2)(ns.hiddenStates)

  -- The length indicators have shape BxL and we replicate it for each hidden
  -- dimension, resulting in BxLxH.
  ns.lenInd = nn.Replicate(hiddenSize,3)(ns.lengthIndicators)

  -- Output layer
  --
  -- We then use the lenInd matrix to mask the output matrix, leaving only 
  -- the terminal vectors for each sequence activated. We can then sum over
  -- the sequence to telescope the matrix, eliminating the L dimension. We
  -- feed this through a linear map to produce the predictions.
  ns.y = nn.Sum(2)((nn.CMulTable()({ns.out, ns.lenInd})))

  -- Combine into a single graph
  local mod = nn.gModule({ns.initial_h, ns.initial_c, ns.inputs, ns.lengthIndicators},{ns.y})

  -- Set up parameter sharing among respective learn maps of each unit. 
  local referenceMaps = ns.linearMaps[1]
  local linearMapsPerMemoryCell = #referenceMaps
  for t=2,maxLength do
    if #ns.linearMaps[t] ~= linearMapsPerMemoryCell then
      error("unexpected number of linear maps: " .. #ns.linearMaps[t])
    end
    for i=1,linearMapsPerMemoryCell do
      local src = referenceMaps[i]
      local map = ns.linearMaps[t][i]
      local srcPar, srcGradPar = src:parameters()
      local mapPar, mapGradPar = map:parameters()
      if #srcPar ~= #mapPar or #srcGradPar ~= #mapGradPar then
        error("parameters structured funny, won't share")
      end
      mapPar[1]:set(srcPar[1])
      mapPar[2]:set(srcPar[2])
      mapGradPar[1]:set(srcGradPar[1])
      mapGradPar[2]:set(srcGradPar[2])
    end
  end

  mod.ns = ns
  return mod
end

return MemoryChain

-- END
