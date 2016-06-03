local ReverseSequence, parent = torch.class('lstm.ReverseSequence', 'nn.Module')

-- This unit takes as input a table containing at least one tensor. One of the
-- dimensions is indicated as the time axis, and this will be reversed. This
-- module works in either batch or non-batch modes. If in batch mode the input
-- must include a second (vector) tensor giving the lengths of the batch members,
-- which may vary.
--
-- It is configured using an integer specifying which dimsion is the time
-- dimension. This is the time axis after already removing the batch dimension,
-- Thus if the shape is BxTxK then it should be ReverseSequence(1) but if
-- the shape is BxKxT then it should be ReverseSequence(2). In non-batch mode,
-- if the shape is TxK then it should be ReverseSequence(1), but if it's
-- KxT then it should be ReverseSequence(2).
--
-- In batch mode:
--
--    lstm.ReverseSequence(timeDim)({inputs,lengths})
--
-- In non-batch mode:
--
--    lstm.ReverseSequence(timeDim)({inputs})
--
-- Components in the inputs tensor that are after the end of the sequence
-- are zeroed out (they should be zero to begin with).
function ReverseSequence:__init(timeDimension)
  parent.__init(self)
  if timeDimension == nil or timeDimension <= 0 then
    error("Invalid timeDimension: " .. timeDimension)
  end
  self.timeDimension = timeDimension
  self.scratchA = torch.Tensor()
  self.scratchB = torch.Tensor()
  self.gradInput = {self.scratchA, self.scratchB}
end

-- This module has no parameters.
function ReverseSequence:parameters()
  return {}
end

function ReverseSequence:updateOutput(tuple)
  local inputs, lengths
  if torch.isTensor(tuple) then
    inputs = tuple
  else
    inputs, lengths = unpack(tuple)
  end
  local batchSize, maxTime
  if lengths == nil then
    -- Non-batch mode
    maxTime = inputs:size(self.timeDimension)
  else
    -- Batch mode
    batchSize = inputs:size(1)
    maxTime = inputs:size(self.timeDimension+1)
  end

  local reversedIndices = torch.range(maxTime, 1, -1):long()
  self.output:resizeAs(inputs):zero()
  -- After picking a batch member, our timeDimension will be one less.
  if batchSize == nil then
    self.output:copy(inputs:index(self.timeDimension, reversedIndices))
  else
    for b=1,batchSize do
      local len = lengths[b]
      local tDim = self.timeDimension
      -- this will be len, len-1, ..., 1
      local trimmedReversedIndices = reversedIndices:narrow(1, maxTime-len+1, len)
      local seqReversed = inputs[b]:narrow(tDim, 1, len):index(tDim, trimmedReversedIndices)
      self.output[b]:narrow(tDim, 1, len):copy(seqReversed)
    end
  end
  return self.output
end

function ReverseSequence:updateGradInput(tuple, upstreamGradOutput)
  local inputs, lengths
  local gradWrtInputs = self.scratchA
  local gradWrtLengths
  if torch.isTensor(tuple) then
    inputs = tuple
    if not torch.isTensor(self.gradInput) then
      self.gradInput = self.scratchA
    end
  else
    inputs, lengths = unpack(tuple)
    if not torch.type(self.gradInput) == "table" then
      self.gradInput = {self.scratchA, self.scratchB}
    end
  end
  if lengths ~= nil then
    -- Batch mode
    batchSize = inputs:size(1)
    gradWrtLengths = self.scratchB
  end
  -- Since reversing is its own opposite, I can just reverse the gradient to get the
  -- gradient of the outputs wrt inputs.
  local gradTuple = {upstreamGradOutput, lengths}
  local reverser = lstm.localize(lstm.ReverseSequence(self.timeDimension))
  local reversedGrad = reverser:forward(gradTuple)
  gradWrtInputs:resizeAs(inputs):zero():copy(reversedGrad)
  if lengths ~= nil then
    gradWrtLengths:resizeAs(lengths):zero()
  end
  return self.gradInput
end

-- END
