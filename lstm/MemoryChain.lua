local stringx = require 'pl.stringx'
local MemoryChain, parent = torch.class('lstm.MemoryChain', 'nn.Module')

-- Constructor. It takes three parameters:
--  1. inputSize [integer] - width of input vector
--  2. hiddenSizes [table of integers] - each entry in the table gives the
--      width of the hidden state for a layer. The number of LSTM layers
--      is determined from the size of this table. The width of the
--      memory cell is the same as the hidden state size for each layer.
--  3. maxLength [integer] - The length of the longest sequence we should
--      expect to see. This is necessary because we pre-create all the
--      memory cells and use the same ones for all sequences.
function MemoryChain:__init(inputSize, hiddenSizes, maxLength)
  print("MemoryChain(" .. inputSize .. ',<' .. stringx.join(',',hiddenSizes) ..
    '>,' .. maxLength .. ')')
  parent.__init(self)
  self.inputSize = inputSize

  self.numLayers = #hiddenSizes
  self.hiddenSizes = hiddenSizes
  -- For convenience, store the inputSize as if it's the 0'th hidden layer
  self.hiddenSizes[0] = inputSize

  self.maxLength = maxLength
  self.gradInput = nil
  -- Later we will cache the batch size and length of the input during the
  -- forward so we don't bother recomputing these during the backward pass.
  self.batchSize = nil
  self.len = nil

  -- We will use the memory associated with the first cell in each layer as the
  -- storage for shared parameters, but we can't share them until later, after
  -- the whole network is created because getParameters will point the tensors
  -- to new storage.
  --
  -- Here I create tables that will store the shared parameters, one for
  -- forward parameters and one for the accumulating gradient estimates.
  self.lstmParams = {}
  self.lstmGradParams = {}

  print("Creating MemoryChain")
  self.lstms = {}
  local prevLayerSize = self.inputSize
  for l=1,self.numLayers do
    local thisLayerSize = self.hiddenSizes[l]
    self.lstms[l] = {}

    -- Make enough lstm cells for the longest sequence
    for t=1,maxLength do
      self.lstms[l][t] = lstm.MemoryCell(prevLayerSize, thisLayerSize)
    end

    -- Capture the parameters of the first cell in this layer, as these will
    -- be shared across the layer.
    self.lstmParams[l], self.lstmGradParams[l] = self.lstms[l][1]:parameters()

    -- The output of one layer is the input to the next.
    prevLayerSize = thisLayerSize
  end
end

-- Return the shared parameters. This function returns two tables, one for
-- forward parameters and one for the accumulating gradient estimates. Each
-- table has an entry for each layer.
function MemoryChain:parameters()
  return nn.FlattenTable():forward(self.lstmParams),
    nn.FlattenTable():forward(self.lstmGradParams)
end

-- Share parameters among all memory cells of each layer. Parameters are not
-- shared between layers.
function MemoryChain:share()
  -- The first cell in each layer is the reference parameters. We'll share all
  -- subsequent cells in the layer back with this first one.
  for l=1,self.numLayers do

    -- Get the shared parameters for this layer.
    local sharedParams = self.lstmParams[l]
    local sharedGradParams = self.lstmGradParams[l]

    for t=2,self.maxLength do
      -- Get the parameters for the memory cell in layer l at timestep t. This
      -- will be a table containing the parameters for each unit in the LSTM
      -- MemoryCell module.
      local cellParams, cellGradParams = self.lstms[l][t]:parameters()

      -- Iterate over each component's parameters, setting them to use the
      -- memory of the reference memory cell, which we captured during setup.
      for i=1, #cellParams do
        cellParams[i]:set(sharedParams[i])
        cellGradParams[i]:set(sharedGradParams[i])
      end
    end
  end
end

-- Set all parameters to uniform on the interval (-radius, radius)
function MemoryChain:reset(radius)
  local par = self:parameters()
  for l=1, self.numLayers do
    local layerPar = par[l]
    -- Iterate over each chunk of parameters (i.e., from each LSTM submodule).
    for i=1, #layerPar do
      layerPar[i]:uniform(-radius, radius)
    end
  end
end

-- Receives a table containing two Tensors: input and a vector of lengths, as
-- not all sequences will span the full length dimension of the tensor. Input
-- should be a 3D tensor with the first dimension iterating over examples in
-- the batch, the second dimension iterating over timesteps in the sequence,
-- and the last dimension iterating over features.
function MemoryChain:updateOutput(tuple)
  local input, lengths = unpack(tuple)
  if input:dim() ~= 3 then
    error("expecting a 3D input")
  end
  self.batchSize = input:size(1)

  -- Cache the length of the longest sequence in this batch (i.e., the size
  -- of the middle dimension). This may be less than maxLength, which is
  -- the longest sequence we will ever see.
  self.len = input:size(2)

  -- We'll use these tables to store the inputs of each memory cell so we have
  -- these for the backwards pass.
  self.hiddenStates = {}
  self.memories = {}
  self.inputs = {}

  -- First forward each layer all the way through the sequence before going
  -- up to the next layer
  for l=1,self.numLayers do
    local thisLayerSize = self.hiddenSizes[l]

    -- The first memory cell in this layer will receive zeros coming from the
    -- left, as there is no memory cell further left.
    local h = lstm.Tensor()(self.batchSize, thisLayerSize):zero()
    local c = lstm.Tensor()(self.batchSize, thisLayerSize):zero()
    self.hiddenStates[l] = {[0] = h}
    self.memories[l] = {[0] = c}
    self.inputs[l] = {}

    -- Iterate over memory cells feeding each successive tuple (h,c) into the next
    -- LSTM memory cell in this layer.
    for t=1,self.len do
      -- If we're in the first layer, we get input from the actual input,
      -- otherwise we get input from the output of the previous layer.
      local x = nil
      if l == 1 then
        x = input:select(2, t)
      else
        x = self.lstms[l-1][t].output[1]
      end
      self.inputs[l][t] = x

      -- Propagate this memory cell forward and prepare get h,c ready for 
      -- the next timestep.
      self.lstms[l][t]:forward({x, h, c})
      h, c = unpack(self.lstms[l][t].output)

      -- Save the hidden states and memories for back-propagation.
      self.hiddenStates[l][t] = h
      self.memories[l][t] = c
    end
  end

  -- Copy the terminal output of the top layer for each batch member into the
  -- output tensor.
  local topLayer = self.numLayers
  local topLayerSize = self.hiddenSizes[topLayer]
  self.output = lstm.Tensor()(self.batchSize, topLayerSize)
  for b=1, self.batchSize do
    h = self.lstms[topLayer][lengths[b]].output[1]
    self.output[b]:copy(h[b])
  end
  return self.output
end

-- This method is the crux of back-propagation. It computes the gradient of the
-- outputs with respect to the inputs.
--
-- `tuple` is the same table that is passed during forward propagation, namely
-- a table containing the inputs and the sequence lengths of the batches.
--
-- `upstreamGradOutput` should be a BxH matrix where B is batch size and H is the
-- hidden state size of the top layer. Each row will correspond to the gradient
-- of the objective function wrt the outputs of the LSTM memory cell at the
-- sequence terminus. However, this isn't necessarily the last memory cell in
-- the `lstms` array because sequences are different lengths.
function MemoryChain:updateGradInput(tuple, upstreamGradOutput)
  -- Rather than use the inputs provided, I use the cached ones from forward
  -- propagation, because I can treate them the same regardless of the layer
  -- we're back-propagating.
  local _, lengths = unpack(tuple)
  local x,h,c

  -- Storage for the gradient wrt inputs of the whole chain.
  self.gradInput = lstm.Tensor()(self.batchSize, self.len, self.inputSize)

  -- Because each batch member has a sequence of a different length less than
  -- or equal to self.len, we need to have some way to propagate errors starting
  -- at the correct level. 
  -- 
  -- I build a binary matrix of size BxL. This matrix will be used to
  -- determine where error terms are propagating back from. The matrix,
  -- terminal, has a one at the terminal column in the sequence. 
  local terminal = lstm.Tensor()(self.batchSize, self.len):zero()
  for b=1,self.batchSize do
    local T = lengths[b]
    terminal[b][T] = 1
  end

  -- Stop at the top layer and work our way back down, computing the gradient
  -- wrt the inputs into the layer.
  local gradOutput
  local topLayer = self.numLayers

  -- Storage for a LSTM cell's gradient signal coming in on the output wires.
  local hUpstream = lstm.Tensor()()
  local cUpstream = lstm.Tensor()()

  for l=topLayer,1,-1 do
    local thisHiddenSize = self.hiddenSizes[l]
    hUpstream:resize(self.batchSize, thisHiddenSize)
    cUpstream:resize(self.batchSize, thisHiddenSize)

    -- Work our way back in time for this layer.
    for t=self.len,1,-1 do
      local currentCell = self.lstms[l][t]

      hUpstream:zero()
      cUpstream:zero()

      x = self.inputs[l][t]
      h = self.hiddenStates[t-1]
      c = self.memories[t-1]

      if l == topLayer then
        -- Replicate our mask of which batch members receive errors from the
        -- upstream gradient at this time step. (replicated hiddenSize times)
        local terminalColumn = terminal:select(2, t):contiguous():view(-1,1)
        local upstreamSelect = torch.mm(terminalColumn, torch.ones(1,thisHiddenSize))
        hUpstream:add(upstreamSelect:cmul(upstreamGradOutput))
      else
        local cellAboveMe = self.lstms[l+1][t]
        hUpstream:add(cellAboveMe.gradInput[1])
      end

      if t < self.len then
        local cellToMyRight = self.lstms[l][t+1]
        hUpstream:add(cellToMyRight.gradInput[2])
        cUpstream:add(cellToMyRight.gradInput[3])
      end

      -- Run the LSTM cell backward
      currentCell:backward({x,h,c}, {hUpstream,cUpstream})

      -- If we're the bottom layer, save gradInput[1] in the gradInput for the
      -- whole chain.
      if l == 1 then
        self.gradInput:select(2,t):copy(currentCell.gradInput[1])
      end
    end
  end
  return self.gradInput
end

-- This happens automatically when calling backward on the individual memory
-- cells in updateGradInput. Not sure what to do about the scale parameter.
function MemoryChain:accGradParameters(input, gradOutput, scale)
end

return MemoryChain

-- END
