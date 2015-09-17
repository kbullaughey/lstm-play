#!/usr/bin/env th

nn = require 'nn'
toy = require '../toy/toy'

-- Allow for command-line control over the seed and number of examples.
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train MLP model using toy data.')
cmd:text()
cmd:text('Options')
cmd:option('-seed',os.time(),'initial random seed (defaults to current time)')
cmd:option('-h1',20,'hidden layer 1 size')
cmd:option('-h2',20,'hidden layer 1 size')
cmd:option('-batch',10,'batch size')
cmd:option('-rate',0.05,'learn rate')
cmd:option('-iter',40,'max number of iterations of SGD')
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- Make it reproduceable
torch.manualSeed(params.seed)

-- Read in the toy model data. This is a Tensor with four columns, the first
-- three are inputs and the last is the targets.
d = torch.load('../toy/fixed_width_3.t7')
N = d:size(1)
num_inputs = d:size(2) - 1

-- Separate data into inputs (x) and targets (y)
x = d:narrow(2, 1, num_inputs)
y = d:narrow(2, num_inputs+1, 1)

-- Decide on the train/test split
test_frac = 0.3
test_n = torch.floor(N * test_frac)
train_n = N - test_n

-- Extract out the training data
x_train = x:narrow(1, 1, train_n)
y_train = y:narrow(1, 1, train_n)

-- Extract out the test data
x_test = x:narrow(1, train_n+1, test_n)
y_test = y:narrow(1, train_n+1, test_n)

-- Normalize the training inputs
norm_mean = x_train:mean()
norm_std = x_train:std()
x_train_n = (x_train - norm_mean) / norm_std

-- Normalize the test inputs according to the training data normalization
-- parameters.
x_test_n = (x_test - norm_mean) / norm_std

-- The nn SGD trainer will need a data structure whereby examples can be accessed
-- via the index operator, [], and which has a size() method.
dataset={};
function dataset:size()
  return torch.floor(train_n / params.batch)
end
for i=1,dataset:size() do 
  local start = (i-1)*params.batch + 1
  dataset[i] = {x_train_n:narrow(1,start,params.batch), y_train:narrow(1,start,params.batch)}
end

-- Set up the neural net
mlp = nn.Sequential()
mlp:add(nn.Linear(num_inputs, params.h1))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(params.h1, params.h2))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(params.h2, 1))

-- Get all the parameters packaged into a vector so we can reset them.
mlp_params = mlp:getParameters()

-- Use least-squares loss function and SGD.
criterion = nn.MSECriterion()

-- Set up a SGD trainer.
trainer = nn.StochasticGradient(mlp, criterion)
trainer.maxIteration = params.iter
trainer.learningRate = params.rate
function trainer:hookIteration(iter)
  print("# test error = " .. criterion:forward(mlp:forward(x_test_n), y_test))
end

-- Train the model, after randomly initializing the parameters and clearing out
-- any existing gradient.
mlp_params:uniform(-0.1, 0.1)
mlp:zeroGradParameters()
print("parameter count: " .. mlp_params:size(1))
print("initial error before training = " .. criterion:forward(mlp:forward(x_test_n), y_test))
trainer:train(dataset)

-- Save the trained model
torch.save('trained_model-2_layer.t7', {mlp=mlp, params=mlp_params})

-- Output predictions along a grid so we can see how well it learned the function. We'll
-- generate inputs without noise so we can see how well it does in the absence of noise,
-- which will give us a sense of whether it's learned the true underlying function.
grid_size = 200
target_grid = torch.linspace(0, toy.max_target, grid_size):view(grid_size,1)
inputs_grid = toy.target_to_inputs(target_grid, 0)
inputs_grid_n = (inputs_grid - norm_mean) / norm_std
predictions = mlp:forward(inputs_grid_n)

-- Use penlight to write the data
pldata = require 'pl.data'
pred_d = pldata.new(predictions:totable())
pred_d:write('grid_predictions-2_layer.csv')

-- END
