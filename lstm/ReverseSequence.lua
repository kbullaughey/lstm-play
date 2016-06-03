local ReverseSequence, parent = torch.class('lstm.ReverseSequence', 'nn.Module')

-- This unit takes as input a table containing at least one tensor. One of the
-- dimensions is indicated as the time axis, and this will be reversed. This
-- module works in either batch or non-batch modes. If in batch mode the input
-- must include a second (vector) tensor giving the lengths of the batch members,
-- which may vary.
--
-- It is configured using an integer specifying which dimsion is the time
-- dimension.
--
-- In batch mode:
--
--    lstm.ReverseSequence(2)({inputs,lengths})
--
-- In non-batch mode:
--
--    lstm.ReverseSequence(2)({inputs})
--
-- Components in the inputs tensor that are after the end of the sequence
-- are zeroed out (they should be zero to begin with).
function ReverseSequence:__init(timeDimension)
  parent.__init(self)
  if timeDimension == nil or timeDimension <= 0 then
    error("Invalid timeDimension: " .. timeDimension)
  end
  self.timeDimension = timeDimension
  self.gradInput = {lstm.Tensor(), lstm.Tensor()}
end

-- This module has no parameters.
function ReverseSequence:parameters()
  return {}
end

function ReverseSequence:updateOutput(tuple)
  local inputs, lengths = unpack(tuple)
  local batchSize
  if lengths ~= nil then
    -- Batch mode
    batchSize = inputs:size(1)
  end

  local maxTime = inputs:size(self.timeDimension)
  local reversedIndices = torch.range(maxTime, 1, -1):long()
  self.output:resizeAs(inputs):zero()
  -- After picking a batch member, our timeDimension will be one less.
  if batchSize == nil then
    self.output:copy(inputs:index(self.timeDimension, reversedIndices))
  else
    for b=1,batchSize do
      local len = lengths[b]
      local tDim = self.timeDimension - 1
      -- this will be len, len-1, ..., 1
      local trimmedReversedIndices = reversedIndices:narrow(1, maxTime-len+1, len)
      local seqReversed = inputs[b]:narrow(tDim, 1, len):index(tDim, trimmedReversedIndices)
      self.output[b]:narrow(tDim, 1, len):copy(seqReversed)
    end
  end
  return self.output
end

function ReverseSequence:updateGradInput(tuple, upstreamGradOutput)
  local inputs, lengths = unpack(tuple)
  if lengths ~= nil then
    -- Batch mode
    batchSize = inputs:size(1)
  end
  -- Since reversing is its own opposite, I can just reverse the gradient to get the
  -- gradient of the outputs wrt inputs.
  local gradTuple = {upstreamGradOutput, lengths}
  local reverser = lstm.localize(lstm.ReverseSequence(self.timeDimension))
  local reversedGrad = reverser:forward(gradTuple)
  self.gradInput[1]:resizeAs(inputs):zero():copy(reversedGrad)
  if lengths ~= nil then
    self.gradInput[2]:resizeAs(lengths):zero()
  end
  return self.gradInput
end

-- END
