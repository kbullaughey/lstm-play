require 'nngraph'

torch.manualSeed(1)
initialPar = torch.Tensor(12):uniform(-0.1, 0.1)
x = torch.Tensor(3):uniform(-1, 1)
y = torch.Tensor(3):uniform(-1, 1)

function makeNet(initialPar)
  local net = {}
  local x = nn.Identity()()
  local y = nn.Linear(3,3)(nn.Linear(3,3)(x))
  local g = nn.gModule({x},{y})
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

function summarize(net, label)
  print("[" .. label .. "] par sum: " .. net.par:sum())
  print("[" .. label .. "] gradPar sum: " .. net.gradPar:sum())
end

a = makeNet(initialPar)
b = makeNet(initialPar)

summarize(a, 'a')
summarize(b, 'b')

roundTrip(a, x, y)
roundTrip(b, x, y)

a.graph:updateParameters(0.05)
b.par:add(b.gradPar * (-0.05))

summarize(a, 'a')
summarize(b, 'b')

-- END
