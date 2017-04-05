require 'torch'
require 'nngraph'

lstm = {}

lstm.MemoryCell = require 'lstm.MemoryCell'
lstm.GRUCell = require 'lstm.GRUCell'
lstm.debugger = require 'lstm.debugger'

require 'lstm.MemoryChain'
require 'lstm.MemoryChainFull'
require 'lstm.MemoryChainDirect'
require 'lstm.MemoryChainDualDirect'
require 'lstm.ReverseSequence'
require 'lstm.ReverseSequenceEven'
require 'lstm.GRUChain'
require 'lstm.GRUChainInitialized'
require 'lstm.GRUChainDirect'
require 'lstm.InvertIndex'
require 'lstm.MixtureGate'
require 'lstm.PartialCrossEntropyCriterion'
require 'lstm.SelectTerminal'
require 'lstm.D'

lstm.NoGrad = require 'lstm.NoGrad'
lstm.RegularizationMask = require 'lstm.RegularizationMask'
lstm.Installer = require 'lstm.Installer'

lstm.ping = function()
  return "pong"
end

local use_cuda = false

-- Enable use of cuda
lstm.cuda = function()
  require 'cutorch'
  require 'cunn'
  use_cuda = true
end

-- Synchronize if usign cuda or just a no-op
lstm.sync = function()
  if use_cuda then
    cutorch.synchronize()
  end
end

-- Choose a tensor constructor
lstm.Tensor = function(...)
  if use_cuda then
    return torch.CudaTensor(...)
  end
  return torch.Tensor(...)
end

lstm.localize = function(thing)
  if not use_cuda then
    return thing
  end
  if torch.isTensor(thing) or type(thing) == "table" then
    if torch.type(thing.cuda) == "function" then
      return thing:cuda()
    end
  end
  return thing
end

function lstm.deepLocalize(t)
  if type(t) ~= 'table' then
    return lstm.localize(t)
  end
  local mt = getmetatable(t)
  local res = {}
  for k,v in pairs(t) do
    res[k] = lstm.deepLocalize(v)
  end
  setmetatable(res,mt)
  return res
end

return lstm
