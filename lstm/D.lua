-- D works like an nn.Identity node, can emit logging information
--
-- It can also permute tensors (if the node takes a single tensor as input).
-- When permuting is enabled, it will permute the first dimension, when
-- enabled, which is assumed to be the batch dimension. The purpose is to screw
-- up the relevance of a node to make sure it is contributing signal to the
-- predictions. It only permute during the forward pass and is not meant to be
-- used during training.
local Class, parent = torch.class('lstm.D', 'nn.Identity')

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

function Class:received(direction, input, grad)
  self:logMessage(direction .. " input: " .. lstm.debugger:describe(input))
  if direction == "backward" then
    self:logMessage(direction .. " gradOutput: " .. lstm.debugger:describe(grad))
  end
end

function Class:__init(label)
  self.label = label or "unlabeled"
  self.shouldPermute = false
  if lstm.debugger.enabled then
    self:logMessage("init")
  end
end

function Class:logMessage(msg)
  print("D[" .. self.label .. "|" .. elapsedTime() .. "]: " .. msg)
end

-- To turn it on, give it a major and minor, to turn it off, give it false
function Class:permute(major, minor)
  if type(major) == "boolean" then
    assert(major == false, "Must pass false to turn permutation off")
    self.shouldPermute = false
  else
    self.major = major
    self.minor = minor
    self.shouldPermute = true
    self.storage = self.storage or lstm.Tensor()
    self.gradStorage = self.gradStorage or lstm.Tensor()
    self.perm = lstm.localize(torch.LongTensor())
  end
  return self
end

function Class:updateOutput(input)
  if lstm.debugger.enabled then
    self:received("forward", input)
  end
  if self.shouldPermute then
    self.storage:resizeAs(input)
    local N = input:size(self.major)
    self.perm:resize(N)
    for i=1,input:size(self.minor) do
      self.perm:randperm(N)
      local permuted = input:select(self.minor,i):index(self.major,self.perm)
      self.storage:select(self.minor,i):copy(permuted)
    end
    self.output = self.storage
  else
    self.output = input
  end
  return self.output
end

function Class:updateGradInput(input, gradOutput)
  if lstm.debugger.enabled then
    self:received("backward", input, gradOutput)
  end
  return parent.updateGradInput(self, input, gradOutput)
end

