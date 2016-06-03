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

function suite.ReverseSequenceForwardBatch()
  local x = torch.Tensor({{1,2,3,0},{4,5,6,7}})
  local lengths = torch.Tensor({3,4})
  local xRev = lstm.ReverseSequence(1):forward({x, lengths})
  local expected = torch.Tensor({{3,2,1,0},{7,6,5,4}})
  tester:eq(xRev, expected, 0, "reversed sequences, batch")
end

function suite.ReverseSequenceBackwardBatch()
  local x = torch.Tensor({{1,2,3,0},{4,5,6,7}})
  local lengths = torch.Tensor({3,4})
  local mod = lstm.ReverseSequence(1)
  local xRev = mod:forward({x, lengths})
  -- Pretend xRev is the gradient. In propagating back it will get reversed.
  local xRecovered = mod:backward({x, lengths}, xRev)
  tester:eq(x, xRecovered[1], 0, "gradient gets reversed, batch")
  tester:eq(torch.zeros(2), xRecovered[2], 0, "grad wrt lengths is zero")
end

function suite.ReverseSequenceEvenForwardBatch()
  local x = torch.Tensor({{1,2,3,4},{4,5,6,7}})
  local xRev = lstm.ReverseSequenceEven(1):forward(x)
  local expected = torch.Tensor({{4,3,2,1},{7,6,5,4}})
  tester:eq(xRev, expected, 0, "reversed sequences, batch")
end

function suite.ReverseSequenceEvenBackwardBatch()
  local x = torch.Tensor({{1,2,3,4},{4,5,6,7}})
  local mod = lstm.ReverseSequenceEven(1)
  local xRev = mod:forward(x)
  -- Pretend xRev is the gradient. In propagating back it will get reversed.
  local xRecovered = mod:backward(x, xRev)
  tester:eq(x, xRecovered, 0, "gradient gets reversed, batch")
end

function suite.ReverseSequenceForwardNonBatchVector()
  local x = torch.Tensor({1,2,3,4})
  -- Check that it works either with a tensor or a table containing a tensor.
  local tries = {{x}, x}
  for _, try in ipairs(tries) do
    local xRev = lstm.ReverseSequence(1):forward(try)
    local expected = torch.Tensor({4,3,2,1})
    tester:eq(xRev, expected, 0, "reversed sequences, vector")
  end
end

function suite.ReverseSequenceBackwardNonBatchVector()
  local x = torch.Tensor({1,2,3,4})
  local mod = lstm.ReverseSequence(1)
  local xRev = mod:forward({x})
  local xRecovered = mod:backward({x}, xRev)
  tester:eq(x, xRecovered[1], 0, "gradient got reversed, vector")
end

function suite.ReverseSequenceBackwardNonBatchVectorNaked()
  local x = torch.Tensor({1,2,3,4})
  local mod = lstm.ReverseSequence(1)
  local xRev = mod:forward(x)
  local xRecovered = mod:backward(x, xRev)
  tester:eq(x, xRecovered, 0, "gradient got reversed, vector")
end

function suite.MemoryChainDirectForwardBatch()
  local maxLen = 4
  local H = 3
  local F = 2
  local B = 5
  local chain = lstm.MemoryChainDirect(F,{H},maxLen)
  local x = torch.rand(B,maxLen,F)
  local lengths = torch.Tensor(B):fill(maxLen)
  local out = chain:forward({x,lengths})
  local expected = torch.LongTensor({B,maxLen,H})
  local outSize = torch.LongTensor(out:size():totable())
  tester:eq(outSize, expected, 0, "forward output has correct size, batch")
end

function suite.MemoryChainDirectForwardNonBatch()
  local maxLen = 4
  local H = 3
  local F = 2
  local chain = lstm.MemoryChainDirect(F,{H},maxLen)
  local x = torch.rand(maxLen,F)
  local out = chain:forward({x})
  local expected = torch.LongTensor({maxLen,H})
  local outSize = torch.LongTensor(out:size():totable())
  tester:eq(outSize, expected, 0, "forward output has correct size, non-batch")
end

function suite.MemoryChainDirectForwardNonBatchNaked()
  local maxLen = 4
  local H = 3
  local F = 2
  local chain = lstm.MemoryChainDirect(F,{H},maxLen)
  local x = torch.rand(maxLen,F)
  local out = chain:forward(x)
  local expected = torch.LongTensor({maxLen,H})
  local outSize = torch.LongTensor(out:size():totable())
  tester:eq(outSize, expected, 0, "forward output has correct size, non-batch")
end

function suite.MemoryChainDirectForwardBackwardBatch()
  local maxLen = 4
  local H = 3
  local F = 2
  local B = 5
  local chain = lstm.MemoryChainDirect(F,{H},maxLen)
  local x = torch.rand(B,maxLen,F)
  local lengths = torch.Tensor(B):fill(maxLen)
  chain:forward({x,lengths})
  local gradAbove = torch.rand(B,maxLen,H)
  local gradBelow = chain:backward({x,lengths}, gradAbove)
  local outSizeInput = torch.LongTensor(gradBelow[1]:size():totable())
  local outSizeLength = torch.LongTensor(gradBelow[2]:size():totable())
  local expOutSizeInput = torch.LongTensor(x:size():totable())
  local expOutSizeLength = torch.LongTensor(lengths:size():totable())
  tester:eq(outSizeInput, expOutSizeInput, 0, "grad wrt inputs has correct size, batch")
  tester:eq(outSizeLength, expOutSizeLength, 0, "grad wrt lengths has correct size, batch")
end

function suite.MemoryChainDirectForwardBackwardNonBatch()
  local maxLen = 4
  local H = 3
  local F = 2
  local chain = lstm.MemoryChainDirect(F,{H},maxLen)
  local x = torch.rand(maxLen,F)
  chain:forward({x})
  local gradAbove = torch.rand(maxLen,H)
  local gradBelow = chain:backward({x}, gradAbove)
  local outSizeInput = torch.LongTensor(gradBelow[1]:size():totable())
  local expOutSizeInput = torch.LongTensor(x:size():totable())
  tester:eq(outSizeInput, expOutSizeInput, 0, "grad wrt inputs has correct size, non-batch")
end

function suite.MemoryChainDirectForwardBackwardNonBatchNaked()
  local maxLen = 4
  local H = 3
  local F = 2
  local chain = lstm.MemoryChainDirect(F,{H},maxLen)
  local x = torch.rand(maxLen,F)
  chain:forward(x)
  local gradAbove = torch.rand(maxLen,H)
  local gradBelow = chain:backward(x, gradAbove)
  tester:eq(torch.isTensor(gradBelow), true)
  local outSizeInput = torch.LongTensor(gradBelow:size():totable())
  local expOutSizeInput = torch.LongTensor(x:size():totable())
  tester:eq(outSizeInput, expOutSizeInput, 0, "grad wrt inputs has correct size, non-batch")
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
