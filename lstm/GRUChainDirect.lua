-- This variant assumes that one wants to do a 1-to-1 sequence labeling task
-- and thus the output will be the hidden states for each timestep.

local stringx = require 'pl.stringx'

local Class, parent = torch.class('lstm.GRUChainDirect', 'lstm.GRUChain')

-- Same as GRUChain, except direct only supports one layer, because it seems
-- to make more sense to interlace forward and backward layers.
function Class:__init(inputSize, hiddenSizes, maxLength, dropout)
  if #hiddenSizes ~= 1 then
    error("GRUChainDirect only works with exactly one layer")
  end
  parent.__init(self, inputSize, hiddenSizes, maxLength, dropout)
end

-- Receives a table containing two Tensors: input and a vector of lengths, as not all
-- sequences will span the full length dimension of the tensor.
-- If input is 3D then the first dimension is batch, otherwise the first dim
-- is the sequence. Last dimension is features.
-- Output is size BxLxH
function Class:updateOutput(tuple)
  local input, lengths = table.unpack(tuple)
  if input:dim() ~= 3 then
    error("expecting a 3D input")
  end
  local batchSize = input:size(1)
  local longestExample = input:size(2)
  lstm.sync()
  local localLengths = lengths:typeAs(torch.Tensor())

  -- Storage for output
  assert(self.numLayers == 1, "Only one layer supported")
  local layerSize = self.hiddenSizes[1]
  self.output:resize(batchSize, longestExample, layerSize):zero()

  local h = self:initialState(1, batchSize)

  -- Iterate over cells feeding each successive h into the next
  -- GRU cell.
  for t=1,longestExample do
    local x
    x = input:select(2, t)
    assert(self.grus[1], "Missing grus")
    assert(self.grus[1][t], "Missing gru at timestep " .. t)
    h = self.grus[1][t]:forward({h, x})
  end

  -- Copy the output for each batch member into the output tensor.
  for b=1, batchSize do
    local batchMemberLength = localLengths[b]
    if batchMemberLength > 0 then
      for t=1,batchMemberLength do
        local unit = self.grus[1][t]
        if unit == nil then
          error("No unit at timestep " .. t .. " for batch member " .. b)
        end
        h = unit.output
        self.output[b][t]:copy(h[b])
      end
    else
      self.output[b]:zero()
    end
  end
  return self.output
end

-- upstreamGradOutput will be a BxLxH matrix where B is batch size L is length
-- and H is hidden state size. It contains the gradient of the objective function
-- wrt outputs from the GRU cell at each position in the sequence.
function Class:updateGradInput(tuple, upstreamGradOutput)
  local input, lengths = table.unpack(tuple)
  local batchSize = input:size(1)
  local len = input:size(2)

  -- Get storage the correct sizes
  self.allGradInput[1]:resize(batchSize, self.hiddenSizes[1]):zero()
  self.allGradInput[2]:resize(batchSize, len, self.inputSize):zero()
  self.allGradInput[3]:resizeAs(lengths):zero()
  lstm.sync()
  local localLengths = lengths:typeAs(torch.Tensor())

  if input:dim() ~= 3 then
    error("GRUChainDirect:updageGradInput is expecting a 3D input tensor")
  end

  -- We actually only have one layer, but this code comes from GRUChain
  assert(self.numLayers == 1, "Only one layer supported")
  local hiddenSize = self.hiddenSizes[1]
  -- Resize scratch memory so it's the right size for this layer.
  self.h:resize(batchSize, hiddenSize)
  self.gradOutputScratch:resize(batchSize, hiddenSize)

  for t=len,1,-1 do
    local gradOutput = self.gradOutputScratch:zero()
    if t == len then
      -- This is the right side. The only gradient we get is from
      -- upstreamGradOutput. The gradOutput for this cell is only non-zero
      -- where there are batch members that teriminate here.
      for b=1,batchSize do
        if localLengths[b] == t then
          gradOutput[b]:add(upstreamGradOutput[b][t])
        end
      end
    else
      -- Not the right-most, so we need to account for messages incoming on the
      -- right.
      local gruRight = self.grus[1][t+1]
      gradOutput = gruRight.gradInput[1]
      -- Only copy messages from above if the batch member is at least this long.
      for b=1,batchSize do
        if localLengths[b] >= t then
          gradOutput[b]:add(upstreamGradOutput[b][t])
        end
      end
    end

    -- Backward propagate this cell
    local x, h
    x = input:select(2,t)
    if t == 1 then
      h = self.h:zero()
    else
      h = self.grus[1][t-1].output
    end
    self.grus[1][t]:backward({h, x}, gradOutput)
    -- We need to update gradInput
    self.allGradInput[2]:select(2, t):copy(self.grus[1][t].gradInput[2])
    if t == 1 then
      self.allGradInput[1]:copy(self.grus[1][1].gradInput[1])
    end
  end
  return self.gradInput
end

-- END
