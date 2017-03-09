local stringx = require 'pl.stringx'

local Class, parent = torch.class('lstm.GRUChain', 'nn.Module')

-- Constructor. It takes three parameters:
--  1. inputSize [integer] - width of input vector
--  2. hiddenSizes [table of integers] - each entry in the table gives the
--      width of the hidden state for a layer. The number of GRU layers
--      is determined from the size of this table. The width of the
--      cell is the same as the hidden state size for each layer.
--  3. maxLength [integer] - The length of the longest sequence we should
--      expect to see. This is necessary because we pre-create all the
--      cells and use the same ones for all sequences.
--  4. dropout [float] - The dropout fraction for recurrent connections.
function Class:__init(inputSize, hiddenSizes, maxLength, dropout)
  print("GRUChain(" .. inputSize .. ',<' .. stringx.join(',',hiddenSizes) ..
    '>,' .. maxLength .. ')')
  parent.__init(self)

  self.inputSize = inputSize
  self.hiddenSizes = hiddenSizes
  self.numLayers = #hiddenSizes

  -- By default we don't care about the gradient wrt the initial state, but if
  -- we're using a GRUChainInitialized, we want that too, so we compute it in case.
  self.allGradInput = {lstm.Tensor(), lstm.Tensor(), lstm.Tensor()}
  self.gradInput = {self.allGradInput[2], self.allGradInput[3]}

  -- There will be enough lstm.GRUCells for sequences of maxLength  but
  -- in any particular batch we will only propagate enough for the current
  -- batch.
  self.maxLength = maxLength
  self.grus = {}
  self.linearMaps = {}
  self.gruDropoutMods = {}

  -- make enough gru cells for the longest sequence
  local inSize = inputSize
  for l=1,self.numLayers do
    local thisHiddenSize = hiddenSizes[l]
    self.grus[l] = {}
    self.linearMaps[l] = {}
    self.gruDropoutMods[l] = {}
    for t=1,maxLength do
      -- Since our original LSTM implementation LSTMChainFull makes one graph out
      -- of all the cells, we use the same call semantics here
      -- which entails passing in h_prev, and x, and getting back h.
      -- We then wrap these into separate graphs, which will be manually 
      -- propagated forward and backward as part of this Module.
      local h_prev = nn.Identity()()
      local x = nn.Identity()()
      local unit, maps, dropoutMod = lstm.GRUCell(h_prev, x, inSize, thisHiddenSize, dropout)
      local gUnit = nn.gModule({h_prev, x}, unit)
      self.grus[l][t] = lstm.localize(gUnit)
      self.linearMaps[l][t] = maps
      self.gruDropoutMods[l][t] = dropoutMod
    end

    inSize = thisHiddenSize
  end
  self:setupSharing()

  -- Create some storage we'll use during forward/backward. We'll resize this once
  -- we know the batch size.
  self.h = lstm.Tensor():typeAs(self.output)
  self.gradOutputScratch = lstm.Tensor():typeAs(self.output)
end

function Class:setupSharing(refMaps)
  refMaps = refMaps or self.linearMaps
  for l=1,self.numLayers do
    -- Set up parameter sharing among respective learn maps of each unit for
    -- this layer. Distinct layers do not share parameters.
    local referenceMaps = refMaps[l][1]
    local linearMapsPerGRUCell = #referenceMaps
    for t=1,self.maxLength do
      if #self.linearMaps[l][t] ~= linearMapsPerGRUCell then
        error("unexpected number of linear maps: " .. #self.linearMaps[l][t])
      end
      -- We ony share a t==1 if we're sharing with another chain's refMaps.
      if refMaps ~= self.linearMaps or t > 1 then
        for i=1,linearMapsPerGRUCell do
          local src = referenceMaps[i]
          local map = self.linearMaps[l][t][i]
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
    end
  end
end

-- Return the parameters for the first unit in each layer, which are the reference
-- sets. All other units in each layer share with these.
function Class:parameters()
  local unitPar = {}
  local unitGradPar = {}
  for l=1,self.numLayers do
    unitPar[l], unitGradPar[l] = self.grus[l][1]:parameters()
  end
  unitPar = nn.FlattenTable():forward(unitPar)
  unitGradPar = nn.FlattenTable():forward(unitGradPar)
  return unitPar, unitGradPar
end

-- The first cell will receive zeros.
function Class:initialState(layer, batchSize)
  local thisHiddenSize = self.hiddenSizes[layer]
  return self.h:resize(batchSize, thisHiddenSize):zero()
end

-- Receives a table containing two Tensors: input and a vector of lengths, as not all
-- sequences will span the full length dimension of the tensor.
-- Input is 3D with the first dimension batch, second dimension is sequence and the
-- last dimenion is features.
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
  self.output:resize(batchSize, topLayerSize):zero()

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

  -- Copy the terminal output of the top layer for each batch member into the
  -- output tensor.
  for b=1, batchSize do
    local batchMemberLength = localLengths[b]
    if batchMemberLength > 0 then
      local unit = self.grus[topLayer][batchMemberLength]
      if unit == nil then
        error("No unit at timestep " .. batchMemberLength .. " for batch member " .. b)
      end
      h = unit.output
      self.output[b]:copy(h[b])
    else
      self.output[b]:zero()
    end
  end
  return self.output
end

-- upstreamGradOutput will be a BxH matrix where B is batch size and H is hidden
-- state size. Each row will correspond to the gradient of the objective function
-- wrt outputs from the GRU cell at the sequence terminus. However, this
-- isn't necessarily the last cell in the `grus` array because sequences
-- are different lengths.
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
    error("GRUChain:updageGradInput is expecting a 3D input tensor")
  end

  -- Because each batch member has a sequence of a different length less than
  -- or equal to self.len, we need to have some way to propagate errors starting
  -- at the correct level. 

  -- Go in reverse order from the highest layer down and from the end back to
  -- the beginning.
  local topLayer = self.numLayers
  for l=topLayer,1,-1 do
    local thisHiddenSize = self.hiddenSizes[l]
    -- Resize scratch memory so it's the right size for this layer.
    self.h:resize(batchSize, thisHiddenSize)
    self.gradOutputScratch:resize(batchSize, thisHiddenSize)

    for t=len,1,-1 do
      local gradOutput
      if t == len then
        if l == topLayer then
          -- This is the top, right corner. The only gradient we get is from
          -- upstreamGradOutput. The gradOutput for this cell is only non-zero
          -- where there are batch members that teriminate here.
          gradOutput = self.gradOutputScratch:zero()
          for b=1,batchSize do
            if localLengths[b] == t then
              gradOutput[b]:add(upstreamGradOutput[b])
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
          -- Only copy messages from above if the batch member terminates here.
          for b=1,batchSize do
            if localLengths[b] == t then
              gradOutput[b]:add(upstreamGradOutput[b])
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

function Class:training()
  parent.training(self)
  for l=1,self.numLayers do
    for _,mod in ipairs(self.gruDropoutMods[l]) do
      mod:training()
    end
  end
end

function Class:evaluate()
  parent.evaluate(self)
  for l=1,self.numLayers do
    for _,mod in ipairs(self.gruDropoutMods[l]) do
      mod:evaluate()
    end
  end
end

-- END
