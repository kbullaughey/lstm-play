#!/usr/bin/env th

toy = require '../toy/toy'
check = require '../scripts/check_gradients'
require 'lstm'

-- Allow for command-line control over the seed and number of examples.
cmd = torch.CmdLine()
cmd:text()
cmd:text('Train single-layer LSTM model using toy data.')
cmd:text()
cmd:text('Options')
cmd:option('-seed',0,'initial random seed (defaults to current time)')
cmd:option('-hidden',16,'hidden state size')
cmd:option('-batch',16,'batch size')
cmd:option('-rate',0.05,'learn rate')
cmd:option('-iter',5,'max number of iterations of SGD')
cmd:option('-trained','trained_model-1_layer-varible-direct.t7','file name for saved trained model')
cmd:option('-grid','grid_predictions-1_layer-variable-direct.csv','file name for saved grid predictions')
cmd:option('-data','../toy/variable_width_2-4-direct.t7','simulated data tensor file')
cmd:option('-mode','train','whether to train or check gradients [train (default) | check]')
cmd:text()

-- parse input params
params = cmd:parse(arg)

print("mode: " .. params.mode)

-- Make it reproducible
torch.manualSeed(params.seed)

-- Read in the toy model data. This is a Tensor with nine columns, the first
-- four are inputs, the next four are outputs and the last is the lengths.
d = torch.load(params.data)
N = d:size(1)
maxLength = (d:size(2) - 1) / 2

-- Separate data into inputs (x) and targets (y)
x = d:narrow(2, 1, maxLength):clone()
y = d:narrow(2, maxLength+1, maxLength):clone()
lengths = d:narrow(2, 2*maxLength+1, 1):clone():view(-1)

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
  v = torch.zeros(maxLength)
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
  for i=1,dataset:size() do 
    local start = (i-1)*batchSize + 1
    local inputs = torch.reshape(x:narrow(1,start,batchSize), batchSize, maxLength, 1)
    local batchLengths = lengths:narrow(1,start,batchSize)
    local targets = y:narrow(1,start,batchSize):select(2,1)
    -- Provide a matrix that masks unused portions of the outputs. This will be
    -- ones over the length of the sequence and zeros afterwards.
    local validSequence = torch.zeros(batchSize,maxLength)
    for b=1,batchSize do
      validSequence[b]:narrow(1,1,batchLengths[b]):fill(1)
    end
    -- Add a zero matrix to every example for the initial h state
    dataset[i] = {{inputs,batchLengths,validSequence}, targets}
  end
  return dataset
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
  local prevGradPar
  
  function trainer:train(dataset)
    local iteration = 1
    local currentLearningRate = self.learningRate
    local module = self.module
    local criterion = self.criterion
  
    local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
  
    local par = module.par
    local gradPar = module.gradPar
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
        if self.exampleHook then
          self:exampleHook(t, iteration, criterion.output)
        end
        prevGradPar = gradPar:clone()
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
    err = err + criterion:forward(net:forward(d[i][1]), d[i][2])
  end
  return err / d:size()
end

trainingDataset = makeDataset(x_train_n, y_train, lengths_train, params.hidden,
  params.batch, maxLength)
testingDataset = makeDataset(x_test_n, y_test, lengths_test, params.hidden,
  params.batch, maxLength)

-- chainIn will be a table like {inputSeq,batchLengths,validSeq}. We'll use validSeq
-- this to constrain predictions to the portions of the sequence we care about in
-- each batch member, given they will have different lengths.
chainIn = nn.Identity()()
inputSeq = nn.SelectTable(1)(chainIn)
batchLengths = nn.SelectTable(2)(chainIn)
validSeq = nn.SelectTable(3)(chainIn)
reversedInputSeq = lstm.ReverseSequence(1)({inputSeq,batchLengths})
chainModBackward = lstm.MemoryChainDirect(1, {params.hidden}, maxLength)
chainModForward = lstm.MemoryChainDirect(1, {params.hidden}, maxLength)
-- Each chain will output tensor of dims BxLxH
chainOutBackward = chainModBackward({reversedInputSeq, batchLengths})
chainOutForward = chainModForward({inputSeq, batchLengths})
chainOutBackwardRereversed = lstm.ReverseSequence(1)({chainOutBackward,batchLengths})

-- Join forward and reverse, resulting in BxLx2H
chainOut = nn.JoinTable(3)({chainOutBackwardRereversed,chainOutForward})
validSeqRep = nn.Replicate(2*params.hidden, 3, 2)(validSeq)
chainOutConstrained = nn.CMulTable()({chainOut,validSeqRep})

-- In order to feed these through the linear map for prediction, we have to
-- reshape so that we have a 2D tensor instead of 3D. Later we reshape it
-- back.
chainOutReshaped = nn.Reshape(params.batch, 2*params.hidden*maxLength)(chainOutConstrained)
predicted = nn.Linear(2*params.hidden*maxLength,1)(chainOutReshaped)
--predictedConstrained = predictedReshaped
net = nn.gModule({chainIn},{predicted})

-- Need to reenable sharing after getParameters(), which broke my sharing.
net.par, net.gradPar = net:getParameters()
chainModForward:setupSharing()
chainModBackward:setupSharing()

-- Use least-squares loss function and SGD.
criterion = nn.MSECriterion()

if params.mode == 'train' then
  trainer = lstmTrainer(net, criterion)
  trainer.maxIteration = params.iter
  trainer.learningRate = params.rate
  function trainer:hookIteration(iter, err)
    print("[" .. iter .. "] current error = " .. err)
    if iter % 2 == 0 then
      print("# test error = " .. averageError(testingDataset))
    end
  end

  print("model parameter count: " .. net.par:size(1))
  print("initial test err = " .. averageError(testingDataset))
  trainer:train(trainingDataset)

  -- Save the trained model
  torch.save(params.trained, {net=net})
  
  -- Output predictions along a grid so we can see how well it learned the function. We'll
  -- generate inputs without noise so we can see how well it does in the absence of noise,
  -- which will give us a sense of whether it's learned the true underlying function.
  grid_size = math.ceil(200/params.batch)*params.batch
  target_grid_orig = torch.linspace(0, toy.max_target, grid_size):view(grid_size,1)
  inputs_grid, target_grid = toy.direct_target_to_inputs(target_grid_orig, 0, maxLength)
  target_grid:select(2,4):fill(0)
  inputs_grid_n = (inputs_grid - norm_mean) / norm_std
  -- Use length 3 for all sequences
  inputs_grid_n:select(2, maxLength):fill(0)
  inputs_grid_lengths = torch.Tensor(grid_size):fill(3)
  inputs_grid_n = torch.reshape(inputs_grid_n, grid_size, maxLength, 1)
  predictionsDataset = makeDataset(inputs_grid_n, target_grid,
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
  exampleLengths = example[1][2]:nonzero():select(2,2)
  -- This method returns a vector containing L ones with the rest zeros.
  local mapRow = function(L)
    v = torch.zeros(4)
    v:narrow(1,1,L):fill(1)
    return v:view(1,-1)
  end
  -- We use mapRow to make a mask matrix so we can zero out inputs that
  -- are not really part of each example.
  mask = nn.JoinTable(1):forward(tablex.map(mapRow, exampleLengths:totable()))
  local get1 = function(z) return z[1] end
  local err = check.checkInputsGrad(net, criterion, example, example[1][1], mask, get1)
  print("error in estimate of inputs Jacobian: " .. err)
  err = check.checkParametersGrad(net, criterion, example, net.par, net.gradPar, get1)
  print("error in estimate of parameters Jacobian: " .. err)

else
  error("invalid mode " .. params.mode)
end
-- END
