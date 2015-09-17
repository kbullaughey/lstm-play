#!/usr/bin/env th

nn = require 'nn'
toy = require './toy'

-- Allow for command-line control over the seed and number of examples.
cmd = torch.CmdLine()
cmd:text()
cmd:text('Simulate from toy model, all examples have 3 inputs')
cmd:text()
cmd:text('Options')
cmd:option('-seed',os.time(),'initial random seed (defaults to current time)')
cmd:option('-n',1000,'number of examples to simulate')
cmd:option('-sd',0.2,'noise standard deviation')
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- Make it reproduceable
torch.manualSeed(params.seed)

-- Simulate n examples. Each example will have three inputs, namely our function
-- valuated at time lags of 0, 1, and 2. i.e. f(target), f(target-1), f(target-2)
n = params.n
target = torch.rand(n,1):mul(toy.max_target)
inputs = toy.target_to_inputs(target, params.sd)

-- Save the data as four columns, targets last, in a t7 file.
data = nn.JoinTable(2):forward({inputs,target})
torch.save('fixed_width_3.t7', data)

-- END
