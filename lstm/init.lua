require 'torch'

lstm = {}

lstm.MemoryCell = require 'lstm.MemoryCell'
require 'lstm.MemoryChain'

lstm.ping = function()
  return "pong"
end

return lstm
