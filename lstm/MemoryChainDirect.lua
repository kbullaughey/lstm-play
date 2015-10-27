-- This variant assumes that one wants to do a 1-to-1 sequence labeling task
-- and thus the output will be the hidden states for each timestep.

local stringx = require 'pl.stringx'

local MemoryChainDirect, parent = torch.class('lstm.MemoryChainDirect', 'lstm.MemoryChain')

-- Same as MemoryChain
function MemoryChainDirect:__init(inputSize, hiddenSizes, maxLength)
  parent.__init(self, inputSize, hiddenSizes, maxLength)
end

-- Receives a table containing two Tensors: input and a vector of lengths, as not all
-- sequences will span the full length dimension of the tensor.
-- If input is 3D then the first dimension is batch, otherwise the first dim
-- is the sequence. Last dimension is features.
-- Output is size BxLxH
function MemoryChainDirect:updateOutput(tuple)
  local input, lengths = unpack(tuple)
  lengths = torch.nonzero(lengths):select(2,2)
  if input:dim() ~= 3 then
    error("expecting a 3D input")
  end
  local batchSize = input:size(1)
  local longestExample = input:size(2)

  -- Storage for output
  local topLayer = self.numLayers
  local topLayerSize = self.hiddenSizes[topLayer]
  self.output:resize(batchSize, longestExample, topLayerSize)

  for l=1, self.numLayers do
    local thisHiddenSize = self.hiddenSizes[l]
    -- The first memory cell will receive zeros.
    local h = self:makeTensor(torch.LongStorage{batchSize,thisHiddenSize})
    local c = self:makeTensor(torch.LongStorage{batchSize,thisHiddenSize})

    -- Iterate over memory cells feeding each successive tuple (h,c) into the next
    -- LSTM memory cell.
    for t=1,longestExample do
      local x
      if l == 1 then
        x = input:select(2, t)
      else
        x = self.lstms[l-1][t].output[1]
      end
      h, c = unpack(self.lstms[l][t]:forward({h, c, x}))
      -- If we're in the top layer, copy h into the output tensor. At present we copy
      -- all timesteps for all batch members. It's up to the prediction layer
      -- to only use the ones that are relevant for each batch memeber.
      if l == topLayer then
        self.output:select(2,t):copy(h)
      end
    end
  end
  return self.output
end

-- upstreamGradOutput will be a BxLxH matrix where B is batch size L is length
-- and H is hidden state size. It contains the gradient of the objective function
-- wrt outputs from the LSTM memory cell at each position in the sequence.
function MemoryChainDirect:updateGradInput(tuple, upstreamGradOutput)
--  print("upstreamGradOuput:sum(3):")
--  print(upstreamGradOutput:sum(3):view(-1,4))
  local input, lengths = unpack(tuple)
  local batchSize = input:size(1)
  local len = input:size(2)
  self.gradInput[1]:resize(batchSize, len, self.inputSize):zero()
  self.gradInput[2]:resizeAs(lengths):zero()

  lengths = torch.nonzero(lengths):select(2,2)
  local h,c
  if input:dim() ~= 3 then
    error("MemoryChainDirect:updageGradInput is expecting a 3D input tensor")
  end

  -- Because each batch member has a sequence of a different length less than
  -- or equal to len, we need to have some way to propagate errors starting
  -- at the correct level. 

  -- Memory we'll use for the upstream messages of each LSTM memory cell.
  -- Since each memory cell outputs an h and c, we need gradients of these.
  local gradOutput = {
    torch.Tensor():typeAs(self.output),
    torch.Tensor():typeAs(self.output)
  }

  -- Go in reverse order from the highest layer down and from the end back to
  -- the beginning.
  --local topLayer = self.numLayers
  local l = 1
  --for l=topLayer,1,-1 do
    local thisHiddenSize = self.hiddenSizes[l]
    gradOutput[1]:resize(batchSize, thisHiddenSize)
    gradOutput[2]:resize(batchSize, thisHiddenSize)
    for t=len,1,-1 do
--      if self.debugHook ~= nil then
--        self.debugHook(t)
--      end
      gradOutput[1]:zero()
      gradOutput[2]:zero()
      -- If we're in the top layer, we'll get some messages from upstreamGradOutput,
      -- otherwise we'll get the messages from the lstm above. In either case, above
      -- will be BxH.
      local above
      --if l == topLayer then
        above = upstreamGradOutput:select(2,t)
--      else
--        local lstmAbove = self.lstms[l+1][t]
--        above = lstmAbove.gradInput[3]
--      end
      -- Only incorporate messages from above if batch member is at least t long.
      for b=1,batchSize do
        if t <= lengths[b] then
          gradOutput[1][b]:add(above[b])
        end
      end
        
      -- Only get messages from the right if we're not at the right-most edge or
      -- this batch member's sequence doesn't extend right.
      if t < len then
        local lstmRight = self.lstms[l][t+1]
        for b=1,batchSize do
          if t < lengths[b] then
            -- message from h
            gradOutput[1][b]:add(lstmRight.gradInput[1][b])
            -- message from c
            gradOutput[2][b]:add(lstmRight.gradInput[2][b])
          end
        end
      end

      -- Backward propagate this memory cell
      local x
      --if l == 1 then
        x = input:select(2,t)
      --else
      --  x = self.lstms[l-1][t].output[1]
      --end
      if t == 1 then
        h = self:makeTensor(torch.LongStorage{batchSize,thisHiddenSize})
        c = self:makeTensor(torch.LongStorage{batchSize,thisHiddenSize})
      else
        h = self.lstms[l][t-1].output[1]
        c = self.lstms[l][t-1].output[2]
      end
--      print("ready to propagate")
--      print(gradOutput[1]:sum(2))
--      print(gradOutput[2]:sum(2))
      self.lstms[l][t]:backward({h, c, x}, gradOutput)
      -- If we're the bottom layer, we need to update gradInput
      --if l == 1 then
        self.gradInput[1]:select(2,t):copy(self.lstms[1][t].gradInput[3])
--        for b=1,batchSize do
--          if t <= lengths[b] then
--            self.gradInput[1][{{b},{t}}]:copy(self.lstms[1][t].gradInput[3][b])
--          end
--        end
      --end
    end
  --end
--  print("gradInput done")
--  print(self.gradInput[1]:view(-1,4))
  return self.gradInput
end

-- END
