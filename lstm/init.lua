require 'torch'

lstm = {}

lstm.MemoryCell = require 'lstm.MemoryCell'
require 'lstm.MemoryChain'
require 'lstm.SelectTerminal'

lstm.ping = function()
  return "pong"
end

local use_cuda = false

-- Enable use of cuda
lstm.cuda = function()
  use_cuda = true
end

-- Choose a tensor constructor
lstm.Tensor = function()
  if use_cuda then
    return torch.CudaTensor
  end
  return torch.Tensor
end

return lstm
