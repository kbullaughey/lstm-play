#!/usr/bin/env th

toy = require '../toy/toy'
require 'nngraph'

-- Allow for command-line control over the seed and number of examples.
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train single-layer LSTM model using toy data.')
cmd:text()
cmd:text('Options')
cmd:option('-seed',os.time(),'initial random seed (defaults to current time)')
cmd:option('-hidden',16,'hidden state size')
cmd:option('-batch',16,'batch size')
cmd:option('-rate',0.05,'learn rate')
cmd:option('-iter',5,'max number of iterations of SGD')
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
  local initial_c = torch.zeros(batchSize, hiddenSize)
  for i=1,dataset:size() do 
    local start = (i-1)*batchSize + 1
    local inputs = torch.reshape(x:narrow(1,start,batchSize), batchSize, numInput, 1)
    local targets = torch.reshape(y:narrow(1,start,batchSize), batchSize, 1, 1)
    -- Add a zero matrix to every example for the initial h state
    dataset[i] = {{initial_h,initial_c,inputs}, targets}
  end
  return dataset
end

LinearMaps = {}
Linear = function(a, b)
  local mod = nn.Linear(a,b)
  table.insert(LinearMaps, mod)
  return mod
end

function addUnit(prev_h, prev_c, x, inputSize, hiddenSize)
  local ns = {}
  -- Input gate. Equation (7)
  ns.i_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h),
    Linear(hiddenSize,hiddenSize)(prev_c)
  }))
  -- Forget gate. Equation (8)
  ns.f_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h),
    Linear(hiddenSize,hiddenSize)(prev_c)
  }))
  -- New contribution to c. Right term in equation (9)
  ns.learning = nn.Tanh()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h)
  }))
  -- Memory cell. Equation (9)
  ns.c = nn.CAddTable()({
    nn.CMulTable()({ns.f_gate, prev_c}),
    nn.CMulTable()({ns.i_gate, ns.learning})
  })
  -- Output gate. Equation (10)
  ns.o_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h),
    Linear(hiddenSize,hiddenSize)(ns.c)
  }))
  -- Updated hidden state. Equation (11)
  ns.h = nn.CMulTable()({ns.o_gate, ns.c})
  return ns
end

-- Build the network
function buildNetwork(inputSize, hiddenSize, length)
  -- Keep a namespace.
  ns = {inputSize=inputSize, hiddenSize=hiddenSize, length=length}
  --local ns = {inputSize=inputSize, hiddenSize=hiddenSize, length=length}

  -- Reset our cache of linear maps.
  LinearMaps = {}

  -- This will be the initial h and c (probably set to zeros)
  ns.initial_h = nn.Identity()()
  ns.initial_c = nn.Identity()()

  -- This will be expecting a matrix of size BxLxI where B is the batch size,
  -- L is the sequence length, and I is the number of inputs.
  ns.inputs = nn.Identity()()
  ns.splitInputs = nn.SplitTable(2)(ns.inputs)
  ns.units = {}

  -- Iterate over the anticipated sequence length, creating a unit for each
  -- timestep.
  local unit = {h=ns.initial_h, c=ns.initial_c}
  for i=1,length do
    local x = nn.SelectTable(i)(ns.splitInputs)
    unit = addUnit(unit.h, unit.c, x, inputSize, hiddenSize)
    ns.units[i] = unit
  end

  -- Output layer
  ns.y = Linear(hiddenSize, 1)(unit.h)

  -- Combine into a single graph
  --local mod = nn.gModule({ns.initial_h, ns.initial_c, ns.inputs},{ns.y})
  mod = nn.gModule({ns.initial_h, ns.initial_c, ns.inputs},{ns.y})

  -- Set up parameter sharing. 
  for t=2,numInput do
    for i=1,11 do
      local src = LinearMaps[i]
      local map = LinearMaps[(t-1)*11+i]
      local srcPar, srcGradPar = src:parameters()
      local mapPar, mapGradPar = map:parameters()
      if #srcPar ~= #mapPar or #srcGradPar ~= #mapGradPar then
        error("parameters structured funny, won't share")
      end
      mapPar[1]:set(srcPar[1])
      mapPar[2]:set(srcPar[2])
      mapGradPar[1]:set(srcGradPar[1])
      mapGradPar[2]:set(srcGradPar[2])
    end
  end

  -- These vectors will be the flattened vectors.
  ns.par, ns.gradPar = mod:getParameters()
  mod.ns = ns
  return mod
end

-- Train the model, based on nn.StochasticGradient
function lstmTrainer(module, criterion)
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

        -- Perform forward propagation on both the lstm to compute the predictions
        -- and on the criterion to compute the error.
        module:forward(input)
        criterion:forward(module.output, target)

        -- Perform back propagation
        criterion:backward(module.output, target)
        module:backward(input, criterion.gradInput)

        currentError = currentError + criterion.output
        module:updateParameters(self.learningRate)
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
    err = err + criterion:forward(net:forward(d[i][1]), d[i][2])
  end
  return err / d:size()
end

trainingDataset = makeDataset(x_train_n, y_train, params.hidden, params.batch)
testingDataset = makeDataset(x_test_n, y_test, params.hidden, params.batch)

net = buildNetwork(1, params.hidden, 3)

-- Use least-squares loss function and SGD.
criterion = nn.MSECriterion()

trainer = lstmTrainer(net, criterion)
trainer.maxIteration = params.iter
trainer.learningRate = params.rate
function trainer:hookIteration(iter, err)
  print("[" .. iter .. "] current error = " .. err)
  if iter % 10 == 0 then
    print("# test error = " .. averageError(testingDataset))
  end
end

print("model parameter count: " .. net.ns.par:size(1))
print("initial test err = " .. averageError(testingDataset))
trainer:train(trainingDataset)

-- Save the trained model
torch.save(params.trained, {net=net})

-- Output predictions along a grid so we can see how well it learned the function. We'll
-- generate inputs without noise so we can see how well it does in the absence of noise,
-- which will give us a sense of whether it's learned the true underlying function.
grid_size = 200
target_grid = torch.linspace(0, toy.max_target, grid_size):view(grid_size,1)
inputs_grid = toy.target_to_inputs(target_grid, 0)
inputs_grid_n = torch.reshape((inputs_grid - norm_mean) / norm_std, grid_size, numInput, 1)
prev_h_grid = torch.zeros(grid_size, params.hidden)
prev_c_grid = torch.zeros(grid_size, params.hidden)
predictions = net:forward({prev_h_grid, prev_c_grid, inputs_grid_n})

-- Use penlight to write the data
pldata = require 'pl.data'
pred_d = pldata.new(predictions:totable())
pred_d:write(params.grid)
-- END
