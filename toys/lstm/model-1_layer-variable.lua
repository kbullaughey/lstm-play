#!/usr/bin/env th

toy = require '../toy/toy'
check = require '../scripts/check_gradients'
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
cmd:option('-trained','trained_model-1_layer-varible.t7','file name for saved trained model')
cmd:option('-grid','grid_predictions-1_layer-variable.csv','file name for saved grid predictions')
cmd:option('-data','../toy/variable_width_2-4-200k.t7','simulated data tensor file')
cmd:option('-mode','train','whether to train or check gradients [train (default) | check]')
cmd:text()

-- parse input params
params = cmd:parse(arg)

print("mode: " .. params.mode)

-- Make it reproducible
torch.manualSeed(params.seed)

-- Read in the toy model data. This is a Tensor with four columns, the first
-- three are inputs and the last is the targets.
d = torch.load(params.data)
N = d:size(1)
maxLength = d:size(2) - 2

-- Separate data into inputs (x) and targets (y)
x = d:narrow(2, 1, maxLength):clone()
lengths = d:narrow(2, maxLength+1, 1):clone():view(-1)
y = d:narrow(2, maxLength+2, 1):clone():view(-1,1)

-- Decide on the train/test split
test_frac = 0.3
test_n = torch.floor(N * test_frac)
train_n = N - test_n

-- Extract out the training data
x_train = x:narrow(1, 1, train_n)
y_train = y:narrow(1, 1, train_n)
lengths_train = lengths:narrow(1, 1, train_n)

-- Extract out the test data
x_test = x:narrow(1, train_n+1, test_n)
y_test = y:narrow(1, train_n+1, test_n)
lengths_test = lengths:narrow(1, train_n+1, test_n)

-- This method returns a vector containing L ones with the rest zeros.
local mapRow = function(L)
  v = torch.zeros(4)
  v:narrow(1,1,L):fill(1)
  return v:view(1,-1)
end
-- We use mapRow to make a mask matrix so we can zero out inputs that
-- are not really part of each example.
mask_train = nn.JoinTable(1):forward(tablex.map(mapRow, lengths_train:totable()))
mask_test = nn.JoinTable(1):forward(tablex.map(mapRow, lengths_test:totable()))

-- Normalize the training inputs
numCells = mask_train:sum()
norm_mean = x_train:sum() / numCells
norm_std = math.sqrt((x_train - norm_mean):cmul(mask_train):pow(2):sum() / numCells)
x_train_n = (x_train - norm_mean) / norm_std

-- Normalize the test inputs according to the training data normalization
-- parameters.
x_test_n = (x_test - norm_mean) / norm_std

x_train_n:cmul(mask_train)
x_test_n:cmul(mask_test)

function makeDataset(x, y, lengths, hiddenSize, batchSize, maxLength)
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
    local inputs = torch.reshape(x:narrow(1,start,batchSize), batchSize, maxLength, 1)
    local targets = torch.reshape(y:narrow(1,start,batchSize), batchSize, 1, 1)
    -- Encode lengths using a one-hot per row strategy
    local batchLengths = torch.zeros(batchSize,maxLength)
    for b=1,batchSize do
      batchLengths[b][lengths:narrow(1,start,batchSize)[b]] = 1
    end
    -- Add a zero matrix to every example for the initial h state
    dataset[i] = {{initial_h,initial_c,inputs,batchLengths}, targets}
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
  ns.i_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h),
    Linear(hiddenSize,hiddenSize)(prev_c)
  }))
  ns.f_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h),
    Linear(hiddenSize,hiddenSize)(prev_c)
  }))
  ns.learning = nn.Tanh()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h)
  }))
  ns.c = nn.CAddTable()({
    nn.CMulTable()({ns.f_gate, prev_c}),
    nn.CMulTable()({ns.i_gate, ns.learning})
  })
  ns.o_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h),
    Linear(hiddenSize,hiddenSize)(ns.c)
  }))
  ns.h = nn.CMulTable()({ns.o_gate, ns.c})
  return ns
end

-- Build the network
function buildNetwork(inputSize, hiddenSize, batchSize, maxLength)
  -- Keep a namespace.
  local ns = {inputSize=inputSize, hiddenSize=hiddenSize, maxLength=maxLength}

  -- Reset our cache of linear maps.
  LinearMaps = {}

  -- This will be the initial h (probably set to zeros)
  ns.initial_h = nn.Identity()()
  ns.initial_c = nn.Identity()()

  -- The length indicators is a one-hot representation of where each sequence
  -- ends. It is a BxL matrix with exactly one 1 in each row.
  ns.lengthIndicators = nn.Identity()()

  -- This will be expecting a matrix of size BxLxI where B is the batch size,
  -- L is the sequence length, and I is the number of inputs.
  ns.inputs = nn.Identity()()
  ns.splitInputs = nn.SplitTable(2)(ns.inputs)

  -- We Save all the hidden states in this table for use in prediction.
  ns.hiddenStates = {}
  ns.hiddenStateMods = {}

  -- Iterate over the anticipated sequence length, creating a unit for each
  -- timestep.
  local unit = {h=ns.initial_h, c=ns.initial_c}
  for t=1,maxLength do
    local x = nn.SelectTable(t)(ns.splitInputs)
    unit = addUnit(unit.h, unit.c, x, inputSize, hiddenSize)
    -- We prepare it to get joined along the time steps as the middle dimension,
    -- by adding an middle dimension with size 1.
    print("timestep " .. t .. ", batchsize: " .. batchSize .. ", hiddenSize " .. hiddenSize)
    ns.hiddenStateMods[t] = nn.Reshape(batchSize, 1, hiddenSize)
    ns.hiddenStates[t] = ns.hiddenStateMods[t](unit.h)
  end

  -- Paste all the hidden matricies together. Each one is BxH and the result
  -- will be BxLxH
  ns.out = nn.JoinTable(2)(ns.hiddenStates)

  -- The length indicators have shape BxL and we replicate it for each hidden
  -- dimension, resulting in BxLxH.
  ns.lenInd = nn.Replicate(hiddenSize,3)(ns.lengthIndicators)

  -- Output layer
  --
  -- We then use the lenInd matrix to mask the output matrix, leaving only 
  -- the terminal vectors for each sequence activated. We can then sum over
  -- the sequence to telescope the matrix, eliminating the L dimension. We
  -- feed this through a linear map to produce the predictions.
  ns.outSelectedMod = nn.Sum(2)
  ns.outSelected = ns.outSelectedMod(nn.CMulTable()({ns.out, ns.lenInd}))
  ns.y = Linear(hiddenSize, 1)(ns.outSelected)

  -- Combine into a single graph
  local mod = nn.gModule({ns.initial_h, ns.initial_c, ns.inputs, ns.lengthIndicators},{ns.y})

  -- Set up parameter sharing. 
  for t=2,maxLength do
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

trainingDataset = makeDataset(x_train_n, y_train, lengths_train, params.hidden,
  params.batch, maxLength)
testingDataset = makeDataset(x_test_n, y_test, lengths_train, params.hidden,
  params.batch, maxLength)

net = buildNetwork(1, params.hidden, params.batch, maxLength)

-- Use least-squares loss function and SGD.
criterion = nn.MSECriterion()

if params.mode == 'train' then
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
  grid_size = math.ceil(200/params.batch)*params.batch
  target_grid = torch.linspace(0, toy.max_target, grid_size):view(grid_size,1)
  inputs_grid = toy.target_to_inputs(target_grid, 0, maxLength)
  inputs_grid_n = (inputs_grid - norm_mean) / norm_std
  -- Use length 3 for all sequences
  inputs_grid_n:select(2, maxLength):fill(0)
  inputs_grid_lengths = torch.Tensor(grid_size):fill(3)
  inputs_grid_n = torch.reshape(inputs_grid_n, grid_size, maxLength, 1)
  predictionsDataset = makeDataset(inputs_grid_n, torch.zeros(grid_size,1),
    inputs_grid_lengths, params.hidden, params.batch, maxLength)
  predictions = {}
  for i=1,predictionsDataset:size() do
    predictions[i] = net:forward(predictionsDataset[i][1]):clone()
  end
  allPredictions = nn.JoinTable(1):forward(predictions)

  -- Use penlight to write the data
  pldata = require 'pl.data'
  pred_d = pldata.new(allPredictions:totable())
  pred_d:write(params.grid)

elseif params.mode == 'check' then

  -- Check gradients for the first training example
  example = trainingDataset[1]
  exampleLengths = example[1][4]:nonzero():select(2,2)
  -- This method returns a vector containing L ones with the rest zeros.
  local mapRow = function(L)
    v = torch.zeros(4)
    v:narrow(1,1,L):fill(1)
    return v:view(1,-1)
  end
  -- We use mapRow to make a mask matrix so we can zero out inputs that
  -- are not really part of each example.
  mask = nn.JoinTable(1):forward(tablex.map(mapRow, exampleLengths:totable()))
  local err = check.checkInputsGrad(net, criterion, example, example[1][3], mask)
  print("error in estimate of inputs Jacobian: " .. err)
  err = check.checkParametersGrad(net, criterion, example, net.ns.par, net.ns.gradPar)
  print("error in estimate of parameters Jacobian: " .. err)

else
  error("invalid mode " .. params.mode)
end
-- END
