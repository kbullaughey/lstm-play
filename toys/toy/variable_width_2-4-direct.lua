#!/usr/bin/env th

-- This toy variant is designed to test MemoryChainDirect and so it is
-- simulating a one-to-one sequence problem.

tablex = require 'pl.tablex'
nn = require 'nn'
toy = require './toy'

-- Allow for command-line control over the seed and number of examples.
cmd = torch.CmdLine()
cmd:text()
cmd:text('Simulate from toy model, examples have length either 2, 3, or 4')
cmd:text()
cmd:text('Options')
cmd:option('-seed',os.time(),'initial random seed (defaults to current time)')
cmd:option('-n',1000,'number of examples to simulate')
cmd:option('-sd',0.2,'noise standard deviation')
cmd:option('-outfile','variable_width_2-4-direct.t7','output file name')
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- Make it reproduceable
--torch.manualSeed(params.seed)
torch.manualSeed(0)

-- Simulate n examples. Each example will have between 2 and 4 inputs, namely our function
-- valuated at time lags of 0, 1, 2, up to 4. i.e.:
--
--    f(target), f(target-1)
--        or
--    f(target), f(target-1), f(target-2)
--        or
--    f(target), f(target-1), f(target-2), f(target-3)
--
n = math.floor(params.n/16)*16
target = torch.rand(n,1):mul(toy.max_target)
inputs, targets = toy.direct_target_to_inputs(target, params.sd, 4)

-- Sample the lengths 2, 3, 4 at prob 1/3 each.
--lengths = torch.rand(n):mul(2):floor():add(3)
lengths = {}
chunk_size = 8
for i=1,n/chunk_size do
  if i % 2 == 0 then
    lengths[i] = torch.Tensor(chunk_size):fill(3)
  else
    lengths[i] = torch.Tensor(chunk_size):fill(4)
  end
end
lengths = nn.JoinTable(1):forward(lengths)

-- This method returns a vector containing L ones with the rest zeros.
local mapRow = function(L)
  v = torch.zeros(4)
  v:narrow(1,1,L):fill(1)
  return v:view(1,-1)
end
-- We use mapRow to make a mask matrix so we can zero out inputs that
-- are not really part of each example.
mask = nn.JoinTable(1):forward(tablex.map(mapRow, lengths:totable()))
inputs:cmul(mask)
targets:cmul(mask)

-- Save the data as nine columns, four columns of inputs, four of targets
-- and one column, last, for lengths. Save as t7 file.
data = nn.JoinTable(2):forward({inputs,targets,lengths:view(-1,1)})
torch.save(params.outfile, data)

-- END
