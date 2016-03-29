local MixtureGate, parent = torch.class('lstm.MixtureGate', 'nn.Module')

-- This unit computs g @ a + (1-g) @ b, where @ is component-wise multiplication.
function MixtureGate:__init()
  parent.__init(self)
  self.gradInput = {torch.Tensor(), torch.Tensor(), torch.Tensor()}
end

-- This module has no parameters.
function MixtureGate:parameters()
  return {}
end

-- This expects a table of three elements, g, a, b
function MixtureGate:updateOutput(tuple)
  local g, a, b = unpack(tuple)
  self.output:resizeAs(g)
  self.output:cmul(g,a)
  self.output:addcmul(-1, g, b):add(1)
  return self.output
end

function MixtureGate:updateGradInput(tuple, upstreamGradOutput)
  local g, a, b = unpack(tuple)
  self.gradInput[1]:resizeAs(g)
  self.gradInput[2]:resizeAs(a)
  self.gradInput[3]:resizeAs(b)
  self.gradInput[1]:add(a, -1, b)
  self.gradInput[2]:copy(g)
  self.gradInput[3]:mul(g, -1)
  return self.gradInput
end

-- END
