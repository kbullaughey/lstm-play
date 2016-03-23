-- D works like an nn.Identity node, can emit logging information
local D, parent = torch.class('lstm.D', 'nn.Identity')

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
  print("D[" .. self.label .. "]: " .. msg)
end

function D:updateOutput(input)
  if lstm.debugger.enabled then
    self:received("forward", input)
  end
  return parent:updateOutput(input)
end

function D:updateGradInput(input, gradOutput)
  if lstm.debugger.enabled then
    self:received("backward", input, gradOutput)
  end
  return parent:updateGradInput(input, gradOutput)
end

