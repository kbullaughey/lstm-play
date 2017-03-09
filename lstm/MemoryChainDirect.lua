-- This variant assumes that one wants to do a 1-to-1 sequence labeling task
-- and thus the output will be the hidden states for each timestep.

local stringx = require 'pl.stringx'

local MemoryChainDirect, parent = torch.class('lstm.MemoryChainDirect', 'lstm.MemoryChain')

-- Same as MemoryChain, except direct only supports one layer, because it seems
-- to make more sense to interlace forward and backward layers.
function MemoryChainDirect:__init(inputSize, hiddenSizes, maxLength)
  if #hiddenSizes ~= 1 then
    error("MemoryChainDirect only works with exactly one layer")
  end
  parent.__init(self, inputSize, hiddenSizes, maxLength)
  -- Save references to these so that if using naked we don't need to allocate new ones.
  self.scratchA = self.gradInput[1]
  self.scratchB = self.gradInput[2]
end

-- In batch mode:
--
--   Receives a table containing two Tensors: input and a vector of lengths, as not all
--   sequences will span the full length dimension of the tensor.
--   The input is 3D with shape BxLxF corresponding to batch, time-axis, and features.
--   Output is size BxLxH, where H is the hidden size.
--
-- In single-sequence mode:
--
--   Receives a table containing a single tensor or just the tensor, input that has shape LxF.
--   Outputs a tensor of shape LxH.
--
function MemoryChainDirect:updateOutput(tuple)
  local input, lengths
  if torch.isTensor(tuple) then
    input = tuple
  else
    input, lengths = table.unpack(tuple)
  end
  local batchSize
  local longestExample
  local h, c
  local layerSize = self.hiddenSizes[1]
  local timeAxis = input:dim() - 1
  if timeAxis == 2 then
    -- Batch mode
    if lengths == nil then
      error("In batch mode must provide lengths")
    end
    batchSize = input:size(1)
    longestExample = input:size(2)
    self.output:resize(batchSize, longestExample, layerSize)
    h = self.h:resize(batchSize,layerSize):zero()
    c = self.c:resize(batchSize,layerSize):zero()
  elseif timeAxis == 1 then
    -- Non-batch mode
    if lengths ~= nil then
      error("In non-batch mode no need to provide lengths")
    end
    longestExample = input:size(1)
    self.output:resize(longestExample, layerSize)
    h = self.h:resize(layerSize):zero()
    c = self.c:resize(layerSize):zero()
  else
    error("Expecting 2D or 3D tensor")
  end
  assert(self.maxLength >= longestExample, "Sequence is too long: " .. longestExample)

  -- Iterate over memory cells feeding each successive tuple (h,c) into the next
  -- LSTM memory cell.
  for t=1,longestExample do
    local x = input:select(timeAxis, t)
    h, c = table.unpack(self.lstms[1][t]:forward({h, c, x}))
    -- At present we copy all timesteps for all batch members. It's up to the
    -- prediction layer to only use the ones that are relevant for each batch
    -- memeber.
    self.output:select(timeAxis,t):copy(h)
  end
  return self.output
end

-- upstreamGradOutput will be a BxLxH matrix where B is batch size L is length
-- and H is hidden state size. It contains the gradient of the objective function
-- wrt outputs from the LSTM memory cell at each position in the sequence.
function MemoryChainDirect:updateGradInput(tuple, upstreamGradOutput)
  local input, lengths
  local gradWrtInputs = self.scratchA
  local gradWrtLengths
  if torch.isTensor(tuple) then
    input = tuple
    if not torch.isTensor(self.gradInput) then
      self.gradInput = self.scratchA
    end
  else
    input, lengths = table.unpack(tuple)
    if not torch.type(self.gradInput) == "table" then
      self.gradInput = {self.scratchA, self.scratchB}
    end
  end
  local len, batchSize
  local timeAxis = input:dim() - 1
  local h,c
  local layerSize = self.hiddenSizes[1]
  if timeAxis == 2 then
    -- Batch mode
    len = input:size(timeAxis)
    batchSize = input:size(1)
    gradWrtInputs:resize(batchSize, len, self.inputSize):zero()
    gradWrtLengths = self.scratchB
    gradWrtLengths:resizeAs(lengths):zero()
    -- Memory we'll use for the upstream messages of each LSTM memory cell.
    -- Since each memory cell outputs an h and c, we need gradients of these.
    self.h:resize(batchSize, layerSize)
    self.c:resize(batchSize, layerSize)
    self.gradOutputScratch.h:resize(batchSize,layerSize)
    self.gradOutputScratch.c:resize(batchSize,layerSize)
  elseif timeAxis == 1 then
    -- Non-batch mode
    len = input:size(timeAxis)
    gradWrtInputs:resize(len, self.inputSize):zero()
    -- Memory we'll use for the upstream messages of each LSTM memory cell.
    -- Since each memory cell outputs an h and c, we need gradients of these.
    self.h:resize(layerSize)
    self.c:resize(layerSize)
    self.gradOutputScratch.h:resize(layerSize)
    self.gradOutputScratch.c:resize(layerSize)
  else
    error("Expecting either a 2D or 3D tensor")
  end

  local gradOutput = {
    self.gradOutputScratch.h, 
    self.gradOutputScratch.c 
  }

  for t=len,1,-1 do
    gradOutput[1]:zero()
    gradOutput[2]:zero()
    -- If we're in the top layer, we'll get some messages from upstreamGradOutput,
    -- otherwise we'll get the messages from the lstm above. In either case, above
    -- will be BxH.
    local above = upstreamGradOutput:select(timeAxis,t)
    -- Only incorporate messages from above if batch member is at least t long.
    if timeAxis == 2 then
      for b=1,batchSize do
        if t <= lengths[b] then
          gradOutput[1][b]:add(above[b])
        end
      end
    else
      -- With only one sequence we know we can use the whole thing.
      gradOutput[1]:add(above)
    end
      
    -- Only get messages from the right if we're not at the right-most edge or
    -- this batch member's sequence doesn't extend right.
    if t < len then
      local lstmRight = self.lstms[1][t+1]
      if timeAxis == 2 then
        for b=1,batchSize do
          if t < lengths[b] then
            -- message from h
            gradOutput[1][b]:add(lstmRight.gradInput[1][b])
            -- message from c
            gradOutput[2][b]:add(lstmRight.gradInput[2][b])
          end
        end
      else
        -- message from h
        gradOutput[1]:add(lstmRight.gradInput[1])
        -- message from c
        gradOutput[2]:add(lstmRight.gradInput[2])
      end
    end

    -- Backward propagate this memory cell
    local x = input:select(timeAxis,t)
    if t == 1 then
      h = self.h:zero()
      c = self.c:zero()
    else
      h = self.lstms[1][t-1].output[1]
      c = self.lstms[1][t-1].output[2]
    end
    self.lstms[1][t]:backward({h, c, x}, gradOutput)
    gradWrtInputs:select(timeAxis,t):copy(self.lstms[1][t].gradInput[3])
  end
  return self.gradInput
end

-- END
