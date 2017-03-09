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
  local topLayer = self.numLayers
  local topLayerSize = self.hiddenSizes[topLayer]
  self.output:resize(batchSize, longestExample, topLayerSize):zero()

  for l=1, self.numLayers do
    local h = self:initialState(l, batchSize)

    -- Iterate over cells feeding each successive h into the next
    -- GRU cell.
    for t=1,longestExample do
      local x
      if l == 1 then
        x = input:select(2, t)
      else
        x = self.grus[l-1][t].output
      end
      assert(self.grus[l], "Missing gru at layer " .. l)
      assert(self.grus[l][t], "Missing gru at layer " .. l .. " timestep " .. t)
      h = self.grus[l][t]:forward({h, x})
    end
  end

  -- Copy the output of the top layer for each batch member into the output tensor.
  for b=1, batchSize do
    local batchMemberLength = localLengths[b]
    if batchMemberLength > 0 then
      for t=1,batchMemberLength do
        local unit = self.grus[topLayer][t]
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
  local topLayer = self.numLayers
  for l=topLayer,1,-1 do
    local thisHiddenSize = self.hiddenSizes[l]
    -- Resize scratch memory so it's the right size for this layer.
    self.h:resize(batchSize, thisHiddenSize)
    self.gradOutputScratch:resize(batchSize, thisHiddenSize)

    for t=len,1,-1 do
      local gradOutput = self.gradOutputScratch:zero()
      if t == len then
        if l == topLayer then
          -- This is the top, right corner. The only gradient we get is from
          -- upstreamGradOutput. The gradOutput for this cell is only non-zero
          -- where there are batch members that teriminate here.
          for b=1,batchSize do
            if localLengths[b] == t then
              gradOutput[b]:add(upstreamGradOutput[b][t])
            end
          end
        else
          -- right edge, but not top layer
          local gruAbove = self.grus[l+1][t]
	        -- Gradient for h
          gradOutput = gruAbove.gradInput[1]
        end
      else
        -- Not the right-most, so we need to account for messages incoming on the
        -- right.
        local gruRight = self.grus[l][t+1]
        gradOutput = gruRight.gradInput[1]
        if l == topLayer then
          -- Only copy messages from above if the batch member is at least this long.
          for b=1,batchSize do
            if localLengths[b] >= t then
              gradOutput[b]:add(upstreamGradOutput[b][t])
            end
          end
        else
          -- Not top layer, so just take grad from above.
          local gruAbove = self.grus[l+1][t]
          -- The h output of this gru is the x input of the one above it.
          gradOutput:add(gruAbove.gradInput[2])
        end
      end

      -- Backward propagate this cell
      local x, h
      if l == 1 then
        x = input:select(2,t)
      else
        x = self.grus[l-1][t].output
      end
      if t == 1 then
        h = self.h:zero()
      else
        h = self.grus[l][t-1].output
      end
      self.grus[l][t]:backward({h, x}, gradOutput)
      -- If we're the bottom layer, we need to update gradInput
      if l == 1 then
        self.allGradInput[2]:select(2, t):copy(self.grus[1][t].gradInput[2])
        if t == 1 then
          self.allGradInput[1]:copy(self.grus[1][1].gradInput[1])
        end
      end
    end
  end
  return self.gradInput
end

-- END
