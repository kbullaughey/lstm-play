require 'torch'
require 'nngraph'

lstm = {}

lstm.MemoryCell = require 'lstm.MemoryCell'
lstm.GRUCell = require 'lstm.GRUCell'
require 'lstm.MemoryChain'
require 'lstm.MemoryChainFull'
require 'lstm.MemoryChainDirect'
require 'lstm.ReverseSequence'
require 'lstm.GRUChain'
require 'lstm.GRUChainDirect'

lstm.ping = function()
  return "pong"
end

local use_cuda = false

-- Enable use of cuda
lstm.cuda = function()
  use_cuda = true
end

-- Synchronize if usign cuda or just a no-op
lstm.sync = function()
  if use_cuda then
    cutorch.synchronize()
  end
end

-- Choose a tensor constructor
lstm.Tensor = function()
  if use_cuda then
    return torch.CudaTensor
  end
  return torch.Tensor
end

lstm.localize = function(thing)
  if use_cuda then
    return thing:cuda()
  end
  return thing
end

return lstm
