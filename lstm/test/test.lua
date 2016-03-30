local lstm = require 'lstm'

local tester = torch.Tester()
local suite = torch.TestSuite()

function suite.ping()
  tester:eq(lstm.ping(), "pong", "ping, pong")
end

-- Assumes input is a table of Tensors. f(x) should be a scalar.
local
finiteDifferenceApprox = function(x, f, h)
  h = h or 1e-5
  local grad = {}
  for i,t in ipairs(x) do
    grad[i] = torch.Tensor():resizeAs(x[i])
    local xv = x[i]:view(-1)
    local gv = grad[i]:view(-1)
    for j=1,xv:size(1) do
      local orig = xv[j]
      local origVal = f(x)
      xv[j] = xv[j] + h
      gv[j] = (f(x) - origVal) / h
      xv[j] = orig
    end
  end
  return grad
end

function suite.MixtureGateExampleForward()
  local mg = lstm.MixtureGate()
  local g = torch.rand(3)
  local a = torch.rand(3)
  local b = torch.rand(3)
  local out = mg:forward({g, a, b})
  local expected = torch.cmul(g, a) + (torch.ones(3)-g):cmul(b)
  tester:eq(out, expected, 1e-4, "forward pass is mixture")
end

function suite.MixtureGateExampleBackwardFD()
  local mg = lstm.MixtureGate()
  local g = torch.rand(3)
  local a = torch.rand(3)
  local b = torch.rand(3)
  local t = torch.rand(3)
  local
  f = function(x)
    return torch.pow(mg:forward(x) - t, 2):mul(0.5):sum()
  end
  local fd = finiteDifferenceApprox({g,a,b}, f)
  local fwd = mg:forward({g,a,b})
  local bkwd = mg:backward({g,a,b}, fwd-t)
  tester:eq(bkwd, fd, 0.01, "finite difference approx. for backprop checks out")
end

function suite.MixtureGateCompareToGraph()
  local input = nn.Identity()()
  local g = nn.SelectTable(1)(input)
  local a = nn.SelectTable(2)(input)
  local b = nn.SelectTable(3)(input)
  local output = nn.CAddTable()({
    -- Cary forward some of the previous hidden state
    nn.CMulTable()({nn.AddConstant(1)(nn.MulConstant(-1)(g)), b}),
    -- Include some of the candidate activation
    nn.CMulTable()({g, a})
  })
  local grph = nn.gModule({input},{output})
  local mg = lstm.MixtureGate()
  local g1 = torch.rand(3)
  local a1 = torch.rand(3)
  local b1 = torch.rand(3)
  local t1 = torch.rand(3)
  local x = {g1, a1, b1}
  local y1 = grph:forward(x)
  local expected = grph:backward(x, y1-t1)
  local y = mg:forward(x)
  local observed = mg:backward(x, y1-t1)
  observed = nn.JoinTable(1):forward(observed)
  expected = nn.JoinTable(1):forward(expected)
  tester:eq(observed, expected, 0.01, "backward is the same as using a graph")
end

tester:add(suite)
tester:run()
