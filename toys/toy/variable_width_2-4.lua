#!/usr/bin/env th

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
cmd:option('-outfile','variable_width_2-4.t7','output file name')
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- Make it reproduceable
torch.manualSeed(params.seed)

-- Simulate n examples. Each example will have between 2 and 4 inputs, namely our function
-- valuated at time lags of 0, 1, 2, up to 4. i.e.:
--
--    f(target), f(target-1)
--        or
--    f(target), f(target-1), f(target-2)
--        or
--    f(target), f(target-1), f(target-2), f(target-3)
--
n = params.n
target = torch.rand(n,1):mul(toy.max_target)
inputs = toy.target_to_inputs(target, params.sd, 4)

-- Sample the lengths 2, 3, 4 at prob 1/3 each.
lengths = torch.rand(n):mul(3):floor():add(2)

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

-- Save the data as five columns, with the last two columns the lengths and
-- targets, in a t7 file.
data = nn.JoinTable(2):forward({inputs,lengths:view(-1,1),target})
torch.save(params.outfile, data)

-- END
