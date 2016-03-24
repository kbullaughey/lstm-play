-- D works like an nn.Identity node, can emit logging information
local D, parent = torch.class('lstm.D', 'nn.Identity')

local lastTime = nil

-- The first call will return 0. All subsequent calls will return the elapsed time since
-- the previos call.
local elapsedTime = function()
  local e = 0
  if lastTime == nil then
    lastTime = torch.tic()
  else
    local newTime = torch.tic()
    e = newTime - lastTime
    lastTime = newTime
  end
  -- return in ms.
  return math.floor(e*1000)
end

function D:received(direction, input, grad)
  self:logMessage(direction .. " input: " .. lstm.debugger:describe(input))
  if direction == "backward" then
    self:logMessage(direction .. " gradOutput: " .. lstm.debugger:describe(grad))
  end
end

function D:__init(label)
  self.label = label or "unlabeled"
  if lstm.debugger.enabled then
    self:logMessage("init")
  end
end

function D:logMessage(msg)
  print("D[" .. self.label .. "|" .. elapsedTime() .. "]: " .. msg)
end

function D:updateOutput(input)
  if lstm.debugger.enabled then
    self:received("forward", input)
  end
  return parent.updateOutput(self, input)
end

function D:updateGradInput(input, gradOutput)
  if lstm.debugger.enabled then
    self:received("backward", input, gradOutput)
  end
  return parent.updateGradInput(self, input, gradOutput)
end

