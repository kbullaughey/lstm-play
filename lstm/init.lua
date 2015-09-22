require 'torch'

lstm = {}

lstm.MemoryCell = require 'lstm.MemoryCell'
lstm.MemoryChain = require 'lstm.MemoryChain'

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
