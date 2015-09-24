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
    '>,' .. batchSize .. "," .. maxLength .. ')')
  parent.__init(self)

  self.inputSize = inputSize
  self.hiddenSizes = hiddenSizes
  self.numLayers = #hiddenSizes

  -- There will be enough lstm.MemoryCells for sequences of maxLength  but
  -- in any particular batch we will only propagate enough for the current
  -- batch.
  self.maxLength = maxLength
  self.lstms = {}

  -- make enough lstm cells for the longest sequence
  local inSize = inputSize
  for l=1,self.numLayers do
    local thisHiddenSize = hiddenSizes[l]
    self.lstms[l] = {}
    local linearMaps = {}
    for t=1,maxLength do
      -- Since our other implementation, MemroyChainFull makes one graph out
      -- of all the memory cells, we need to adapt the same call semantics here
      -- which entails passing in h_prev, c_prev, and x, and getting back h,c.
      -- We then wrap these into separate graphs, which will be manually 
      -- propagated forward and backward as part of this Module.
      local h_prev = nn.Identity()()
      local c_prev = nn.Identity()()
      local x = nn.Identity()()
      local unit, maps = lstm.MemoryCell(h_prev, c_prev, x, inSize, thisHiddenSize)
      local gUnit = nn.gModule({h_prev, c_prev, x},unit)
      self.lstms[l][t] = gUnit
      linearMaps[t] = maps
    end

    -- Set up parameter sharing among respective learn maps of each unit for
    -- this layer. Distinct layers do not share parameters.
    local referenceMaps = linearMaps[1]
    local linearMapsPerMemoryCell = #referenceMaps
    for t=2,maxLength do
      if #linearMaps[t] ~= linearMapsPerMemoryCell then
        error("unexpected number of linear maps: " .. #linearMaps[t])
      end
      for i=1,linearMapsPerMemoryCell do
        local src = referenceMaps[i]
        local map = linearMaps[t][i]
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
    inSize = thisHiddenSize
  end
end

-- Return the parameters for the first unit in each layer, which are the reference
-- sets. All other units in each layer share with these.
function MemoryChain:parameters()
  local unitPar = {}
  local unitGradPar = {}
  for l=1,self.numLayers do
    unitPar[l], unitGradPar[l] = self.lstms[l][1]:parameters()
  end
  unitPar = nn.FlattenTable():forward(unitPar)
  unitGradPar = nn.FlattenTable():forward(unitGradPar)
  return unitPar, unitGradPar
end

-- Convenience method for making tensors that match the type of self.output
function MemoryChain:makeTensor(a,b,c,d)
  return torch.Tensor().typeAs(self.output):resize(a,b,c,d):zero()
end

-- Receives a table containing two Tensors: input and a vector of lengths, as not all
-- sequences will span the full length dimension of the tensor.
-- If input is 3D then the first dimension is batch, otherwise the first dim
-- is the sequence. Last dimension is features.
function MemoryChain:updateOutput(tuple)
  local input, lengths = unpack(tuple)
  if input:dim() ~= 3 then
    error("expecting a 3D input")
  end
  local batchSize = input:size(1)
  local longestExample = input:size(2)

  -- Storage for output
  local topLayer = self.numLayers
  local topLayerSize = self.hiddenSizes[topLayer]
  self.output:resize(batchSize, topLayerSize)

  for l=1, this.numLayers do
    local thisHiddenSize = self.hiddenSizes[l]
    -- The first memory cell will receive zeros.
    local h = self:makeTensor(batchSize, thisHiddenSize)
    local c = self:makeTensor(batchSize, thisHiddenSize)

    -- Iterate over memory cells feeding each successive tuple (h,c) into the next
    -- LSTM memory cell.
    for t=1,longestExample do
      local x
      if l == 1 then
        x = input:select(2, x)
      else
        x = self.lstms[l-1][t].output[1]
      end
      h, c = unpack(self.lstms[l][t]:forward({x, h, c}))
    end
  end

  -- Copy the terminal output of the top layer for each batch member into the
  -- output tensor.
  for b=1, batchSize do
    local batchMemberLength = lengths[b]
    h = self.lstms[topLayer][batchMemberLength].output[1]
    self.output[b]:copy(h[b])
  end
  return self.output
end

-- upstreamGradOutput will be a BxH matrix where B is batch size and H is hidden
-- state size. Each row will correspond to the gradient of the objective function
-- wrt outputs from the LSTM memory cell at the sequence terminus. However, this
-- isn't necessarily the last memory cell in the `lstms` array because sequences
-- are different lengths.
function MemoryChain:updateGradInput(tuple, upstreamGradOutput)
  local input, lengths = unpack(tuple)
  local h,c
  if input:dim() ~= 3 then
    error("MemoryChain:updageGradInput is expecting a 3D input tensor")
  end

  local batchSize = input:size(1)
  local len = input:size(2)
  self.gradInput.resize(batchSize, len, self.inputSize):zero()

  -- Because each batch member has a sequence of a different length less than
  -- or equal to self.len, we need to have some way to propagate errors starting
  -- at the correct level. 

  -- Go in reverse order from the highest layer down and from the end back to
  -- the beginning.
  local topLayer = self.numLayers
  for l=topLayer,1,-1 do
    local thisHiddenSize = self.hiddenSizes[l]
    for t=len,1,-1 do
      local gradOutput
      if t == len then
        if l == topLayer then
          -- This is the top, right corner. The only gradient we get is from
          -- upstreamGradOutput. The gradOutput for this cell is only non-zero
          -- where there are batch members that teriminate here.
          local gradOutput = {
            -- Gradient for h
            torch.Tensor():typeAs(self.output):resize(batchSize, thisHiddenSize):zero()
            -- Gradient for c
            torch.Tensor():typeAs(self.output):resize(batchSize, thisHiddenSize):zero()
          }
          for b=1,batchSize do
            if lengths[b] == t then
              gradOutput[1][b]:add(upstreamGradOutput[b])
            end
          end
        else
          -- right edge, but not top layer
          local lstmAbove = self.lstms[l+1][t]
          gradOutput = {
            -- Gradient for h
            lstmAbove.gradInput[1]
            -- Gradient for c
            torch.Tensor():typeAs(self.output):resize(batchSize, thisHiddenSize):zero()
          }
        end
      else
        -- Not the right-most, so we need to account for messages incoming on the
        -- right.
        local lstmRight = self.lstms[l][t+1]
        gradOutput = { lstmRight.gradInput[1], lstmRight.gradInput[2] }
        if l == topLayer then
          -- Only copy messages from above if the batch member terminates here.
          for b=1,batchSize do
            if lengths[b] == t then
              gradOutput[1][b]:add(upstreamGradOutput[b])
            end
          end
        else
          -- Not top layer, so just take grad from above.
          local lstmAbove = self.lstms[l+1][t]
          gradOutput[1]:add(lstmAbove.gradInput[1]) -- h
          gradOutput[2]:add(lstmAbove.gradInput[2]) -- c
        end
      end

      -- Backward propagate this memory cell
      local x
      if l == 1 then
        x = input:select(2,t)
      else
        x = self.lstms[l-1][t].output[1]
      end
      if t == 1 then
        h = self:makeTensor(batchSize, thisHiddenSize)
        c = self:makeTensor(batchSize, thisHiddenSize)
      else
        h = self.lstms[l][t-1].output[1]
        c = self.lstms[l][t-1].output[2]
      end
      self.lstms[l][t]:backward({h, c, x}, gradOutput)
      -- If we're the bottom layer, we need to update gradInput
      if l == 1 then
        self.gradInput:select(2, t):copy(self.lstms[1][t].gradInput[3])
      end
    end
  end
  return self.gradInput
end

return MemoryChain

-- END
