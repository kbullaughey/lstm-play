#!/usr/bin/env th

check = require './check_gradients'
require 'lstm'

mod = lstm.ReverseSequence(1)
x = torch.rand(4,3)
y = torch.rand(4,3)
x[1][3] = 0
x[4][3] = 0
y[1][3] = 0
y[4][3] = 0
lens = torch.Tensor{2,3,3,2}
criterion = nn.MSECriterion()

-- Check gradients for the first training example
example = {{x,lens},y}

-- This method returns a vector containing L ones with the rest zeros.
local mapRow = function(L)
  v = torch.zeros(3)
  v:narrow(1,1,L):fill(1)
  return v:view(1,-1)
end
-- We use mapRow to make a mask matrix so we can zero out inputs that
-- are not really part of each example.
mask = nn.JoinTable(1):forward(tablex.map(mapRow, lens:totable()))
local extract = function(gradInput)
  return gradInput[1]
end
local err = check.checkInputsGrad(mod, criterion, example, example[1][1], mask, extract)
print("error in estimate of inputs Jacobian: " .. err)

-- END
