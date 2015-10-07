#!/usr/bin/env th

-- This is an RNN model with one hidden layer. It consists of three units,
-- each of which expects one input and the units are recursively connected.

require 'nngraph'
toy = require '../toy/toy'

-- Allow for command-line control over the seed and number of examples.
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train single-layer RNN model using toy data.')
cmd:text()
cmd:text('Options')
cmd:option('-seed',os.time(),'initial random seed (defaults to current time)')
cmd:option('-hidden',26,'hidden state size')
cmd:option('-batch',10,'batch size')
cmd:option('-rate',0.05,'learn rate')
cmd:option('-iter',100,'max number of iterations of SGD')
cmd:option('-trained','trained_model-1_layer.t7','file name for saved trained model')
cmd:option('-grid','grid_predictions-1_layer.csv','file name for saved grid predictions')
cmd:text()

-- parse input params
params = cmd:parse(arg)

-- Make it reproducible
torch.manualSeed(params.seed)

-- Read in the toy model data. This is a Tensor with four columns, the first
-- three are inputs and the last is the targets.
d = torch.load('../toy/fixed_width_3.t7')
N = d:size(1)
numInput = d:size(2) - 1

-- Separate data into inputs (x) and targets (y)
x = d:narrow(2, 1, numInput)
y = d:narrow(2, numInput+1, 1)

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

function makeDataset(x, y, hiddenSize, batchSize)
  dataset={batchSize=batchSize};
  local n = torch.floor(x:size(1) / batchSize)
  -- The nn SGD trainer will need a data structure whereby examples can be accessed
  -- via the index operator, [], and which has a size() method.
  function dataset:size()
    return n
  end
  local initial_h = torch.zeros(batchSize, hiddenSize)
  for i=1,dataset:size() do 
    local start = (i-1)*batchSize + 1
    local inputs = x:narrow(1,start,batchSize)
    local targets = y:narrow(1,start,batchSize)
    -- Add a zero matrix to every example for the initial h state
    dataset[i] = {{initial_h,inputs}, targets}
  end
  return dataset
end

function addUnit(prev_h, x, inputSize, hiddenSize)
  local ns = {}
  -- Concatenate x and prev_h into one input matrix. x is a Bx1 vector and
  -- prev_h is a BxH vector where B is batch size and H is hidden size.
  ns.phx = nn.JoinTable(2,2)({prev_h,x})
  -- Feed these through a combined linear map and squash it.
  ns.h = nn.Tanh()(nn.Linear(inputSize+hiddenSize, hiddenSize)({ns.phx}))
  return ns
end

-- Build the network
function buildNetwork(inputSize, hiddenSize, length)
  -- Keep a namespace.
  local ns = {inputSize=inputSize, hiddenSize=hiddenSize, length=length}

  -- This will be the initial h (probably set to zeros)
  ns.initial_h = nn.Identity()()

  -- This will be expecting a matrix of size BxLxI where B is the batch size,
  -- L is the sequence length, and I is the number of inputs.
  ns.inputs = nn.Identity()()
  ns.splitInputs = nn.SplitTable(2)(ns.inputs)

  -- Iterate over the anticipated sequence length, creating a unit for each
  -- timestep.
  local unit = {h=ns.initial_h}
  for i=1,length do
    local x = nn.Reshape(1)(nn.SelectTable(i)(ns.splitInputs))
    unit = addUnit(unit.h, x, inputSize, hiddenSize)
  end

  -- Output layer
  ns.y = nn.Linear(hiddenSize, 1)(unit.h)

  -- Combine into a single graph
  local mod = nn.gModule({ns.initial_h, ns.inputs},{ns.y})

  -- Set up parameter sharing. The parameter tables will each contain 2L+2
  -- tensors. Each of the first L pairs of tensors will be the linear map
  -- matrix and bias vector for the recurrent units. We'll link each of
  -- these back to the corresponding first one. 
  ns.paramsTable, ns.gradParamsTable = mod:parameters()
  for t=2,length do
    -- Share weights matrix
    ns.paramsTable[2*t-1]:set(ns.paramsTable[1])
    -- Share bias vector
    ns.paramsTable[2*t]:set(ns.paramsTable[2])
    -- Share weights matrix gradient estimate
    ns.gradParamsTable[2*t-1]:set(ns.gradParamsTable[1])
    -- Share bias vector gradient estimate
    ns.gradParamsTable[2*t]:set(ns.gradParamsTable[2])
  end

  -- These vectors will be the flattened vectors.
  ns.par, ns.gradPar = mod:getParameters()
  mod.ns = ns
  return mod
end

-- Train the model, based on nn.StochasticGradient
function rnnTrainer(module, criterion)
  local trainer = {}
  trainer.learningRate = 0.01
  trainer.learningRateDecay = 0
  trainer.maxIteration = 25
  trainer.module = module
  trainer.criterion = criterion
  trainer.verbose = true
  
  function trainer:train(dataset)
    local iteration = 1
    local currentLearningRate = self.learningRate
    local module = self.module
    local criterion = self.criterion
  
    local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
  
    local par = module.ns.par
    local gradPar = module.ns.gradPar
    par:uniform(-0.02, 0.02)

    while true do
      local currentError = 0
      local batches = 0
      for t = 1,dataset:size() do
        local example = dataset[shuffledIndices[t]]
        local input = example[1]
        local target = example[2]
        gradPar:zero()

        -- Perform forward propagation on both the rnn to compute the predictions
        -- and on the criterion to compute the error.
        module:forward(input)
        criterion:forward(module.output, target)

        -- Perform back propagation
        criterion:backward(module.output, target)
        module:backward(input, criterion.gradInput)

        currentError = currentError + criterion.output
        par:add(-self.learningRate, gradPar)
        batches = batches + 1
        if t % 1000 == 0 then
          print ("current partial error = " .. currentError / batches)
        end
      end
      currentError = currentError / dataset:size()
  
      if self.hookIteration then
        self.hookIteration(self, iteration, currentError)
      end
  
      iteration = iteration + 1
      currentLearningRate = self.learningRate / (1 + iteration * self.learningRateDecay)
      if self.maxIteration > 0 and iteration > self.maxIteration then
        print("# you have reached the maximum number of iterations")
        print("# training error = " .. currentError)
        break
      end
    end
  end
  return trainer
end

function averageError(d)
  local err = 0
  for i=1,d:size() do
    err = err + criterion:forward(rnn:forward(d[i][1]), d[i][2])
  end
  return err / d:size()
end

trainingDataset = makeDataset(x_train_n, y_train, params.hidden, params.batch)
testingDataset = makeDataset(x_test_n, y_test, params.hidden, params.batch)

rnn = buildNetwork(1, params.hidden, 3)

-- Use least-squares loss function and SGD.
criterion = nn.MSECriterion()

trainer = rnnTrainer(rnn, criterion)
trainer.maxIteration = params.iter
trainer.learningRate = params.rate
function trainer:hookIteration(iter, err)
  print("[" .. iter .. "] current error = " .. err)
  if iter % 10 == 0 then
    print("# test error = " .. averageError(testingDataset))
  end
end

print("model parameter count: " .. rnn.ns.par:size(1))
print("initial test err = " .. averageError(testingDataset))
trainer:train(trainingDataset)

-- Save the trained model
torch.save(params.trained, {rnn=rnn})

-- Output predictions along a grid so we can see how well it learned the function. We'll
-- generate inputs without noise so we can see how well it does in the absence of noise,
-- which will give us a sense of whether it's learned the true underlying function.
grid_size = 200
target_grid = torch.linspace(0, toy.max_target, grid_size):view(grid_size,1)
inputs_grid = toy.target_to_inputs(target_grid, 0)
inputs_grid_n = (inputs_grid - norm_mean) / norm_std
predictions = rnn:forward({torch.zeros(grid_size,params.hidden), inputs_grid_n})

-- Use penlight to write the data
pldata = require 'pl.data'
pred_d = pldata.new(predictions:totable())
pred_d:write(params.grid)
-- END
