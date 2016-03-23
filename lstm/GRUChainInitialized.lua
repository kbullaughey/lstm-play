local Class, parent = torch.class('lstm.GRUChainInitialized', 'lstm.GRUChain')

function Class:__init(inputSize, hiddenSizes, maxLength)
  parent:__init(inputSize, hiddenSizes, maxLength)
  self.gradInput = self.allGradInput
end

-- Only the first layer can receive an initial state.
function Class:initialState(layer, batchSize)
  if layer > 1 then
    return parent:initialState(layer, batchSize)
  end
  return self.initial
end

-- Receives a table containing three Tensors: initial state, input and a vector of
-- lengths, as not all sequences will span the full length dimension of the tensor.
-- If input is 3D then the first dimension is batch, otherwise the first dim
-- is the sequence. Last dimension is features.
function Class:updateOutput(tuple)
  local initial, input, lengths = unpack(tuple)
  self.initial = initial
  return parent:updateOutput({input,lengths})
end

function Class:updateGradInput(tuple, upstreamGradOutput)
  local initial, input, lengths = unpack(tuple)
  parent:updateGradInput({input,lengths}, upstreamGradOutput)
  return self.allGradInput
end
