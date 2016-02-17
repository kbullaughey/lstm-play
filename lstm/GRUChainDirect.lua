-- This variant assumes that one wants to do a 1-to-1 sequence labeling task
-- and thus the output will be the hidden states for each timestep.

local stringx = require 'pl.stringx'

local GRUChainDirect, parent = torch.class('lstm.GRUChainDirect', 'lstm.GRUChain')

-- Same as GRUChain, except direct only supports one layer, because it seems
-- to make more sense to interlace forward and backward layers.
function GRUChainDirect:__init(inputSize, hiddenSizes, maxLength)
  if #hiddenSizes ~= 1 then
    error("GRUChainDirect only works with exactly one layer")
  end
  parent.__init(self, inputSize, hiddenSizes, maxLength)
end

-- Receives a table containing two Tensors: input and a vector of lengths, as not all
-- sequences will span the full length dimension of the tensor.
-- If input is 3D then the first dimension is batch, otherwise the first dim
-- is the sequence. Last dimension is features.
-- Output is size BxLxH
function GRUChainDirect:updateOutput(tuple)
  local input, lengths = unpack(tuple)
  if input:dim() ~= 3 then
    error("expecting a 3D input")
  end
  local batchSize = input:size(1)
  local longestExample = input:size(2)

  -- Storage for output
  local layerSize = self.hiddenSizes[1]
  self.output:resize(batchSize, longestExample, layerSize)

  -- The first gru cell will receive zeros.
  local h = self.h:resize(batchSize,layerSize):zero()

  -- Iterate over gru cells feeding each successive tuple h into the next
  -- GRU cell.
  for t=1,longestExample do
    local x = input:select(2, t)
    h = self.grus[1][t]:forward({h, x})
    -- At present we copy all timesteps for all batch members. It's up to the
    -- prediction layer to only use the ones that are relevant for each batch
    -- memeber.
    self.output:select(2,t):copy(h)
  end
  return self.output
end

-- upstreamGradOutput will be a BxLxH matrix where B is batch size L is length
-- and H is hidden state size. It contains the gradient of the objective function
-- wrt outputs from the GRU cell at each position in the sequence.
function GRUChainDirect:updateGradInput(tuple, upstreamGradOutput)
  local input, lengths = unpack(tuple)
  local batchSize = input:size(1)
  local len = input:size(2)
  self.gradInput[1]:resize(batchSize, len, self.inputSize):zero()
  self.gradInput[2]:resizeAs(lengths):zero()

  local h
  if input:dim() ~= 3 then
    error("GRUChainDirect:updageGradInput is expecting a 3D input tensor")
  end

  -- Because each batch member has a sequence of a different length less than
  -- or equal to len, we need to have some way to propagate errors starting
  -- at the correct level. 

  local layerSize = self.hiddenSizes[1]

  -- Memory we'll use for the upstream messages of each GRU cell.
  -- Since each gru cell outputs an h and c, we need gradients of these.
  self.h:resize(batchSize, layerSize)
  self.gradOutputScratch:resize(batchSize,layerSize)
  local gradOutput = self.gradOutputScratch

  for t=len,1,-1 do
    gradOutput:zero()
    -- If we're in the top layer, we'll get some messages from upstreamGradOutput,
    -- otherwise we'll get the messages from the gru above. In either case, above
    -- will be BxH.
    local above = upstreamGradOutput:select(2,t)
    -- Only incorporate messages from above if batch member is at least t long.
    for b=1,batchSize do
      if t <= lengths[b] then
        gradOutput[b]:add(above[b])
      end
    end
      
    -- Only get messages from the right if we're not at the right-most edge or
    -- this batch member's sequence doesn't extend right.
    if t < len then
      local gruRight = self.grus[1][t+1]
      for b=1,batchSize do
        if t < lengths[b] then
          -- message from h
          gradOutput[b]:add(gruRight.gradInput[1][b])
        end
      end
    end

    -- Backward propagate this gru cell
    local x = input:select(2,t)
    if t == 1 then
      h = self.h:zero()
    else
      h = self.grus[1][t-1].output
    end
    self.grus[1][t]:backward({h, x}, gradOutput)
    self.gradInput[1]:select(2,t):copy(self.grus[1][t].gradInput[2])
  end
  return self.gradInput
end

-- END
