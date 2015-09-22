require 'nn'

torch.manualSeed(1)
initialPar = torch.Tensor(12):uniform(-0.1, 0.1)
x = torch.Tensor(3):uniform(-1, 1)
y = torch.Tensor(3):uniform(-1, 1)
outputLine = 1

function makeNet(initialPar)
  local net = {}
  -- I create a simple network that consists of two 3x3 linear maps.
  local g = nn.Sequential()
  g:add(nn.Linear(3,3))
  g:add(nn.Linear(3,3))
  -- Tie parameters of two linear maps together
  local parTensors, gradParTensors = g:parameters()
  parTensors[3]:set(parTensors[1])
  parTensors[4]:set(parTensors[2])
  gradParTensors[3]:set(gradParTensors[1])
  gradParTensors[4]:set(gradParTensors[2])

  net.graph = g
  net.par, net.gradPar = g:getParameters()
  net.par:copy(initialPar)
  net.graph:zeroGradParameters()
  net.criterion = nn.MSECriterion()
  return net
end

function roundTrip(net, x,y)
  net.graph:forward(x)
  net.criterion:forward(net.graph.output, y)
  net.criterion:backward(net.graph.output, y)
  net.graph:backward(x, net.criterion.gradInput)
end

function log(s)
  print(outputLine .. ": " .. s)
  outputLine = outputLine + 1
end

function summarize(net, label)
  log("[" .. label .. "] par sum: " .. net.par:sum() .. "; gradPar sum: " .. net.gradPar:sum())
end

a = makeNet(initialPar)
b = makeNet(initialPar)

summarize(a, 'a')
summarize(b, 'b')

roundTrip(a, x, y)
roundTrip(b, x, y)

summarize(a, 'a')
summarize(b, 'b')

a.graph:updateParameters(0.05)
b.par:add(b.gradPar * (-0.05))

summarize(a, 'a')
summarize(b, 'b')

-- END
