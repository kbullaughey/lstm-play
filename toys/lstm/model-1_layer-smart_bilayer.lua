#!/usr/bin/env th

-- TODO:
-- get rid of OneHot term in dataset

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
cmd:option('-iter',4,'max number of iterations of SGD')
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
maxLength = 4

-- Separate data into inputs (x) and targets (y)
x = d:narrow(2, 1, maxLength):clone()
y = d:narrow(2, 5, maxLength):clone()
x = x:reshape(N, maxLength, 1)
lengths = d:select(2,9)

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
norm_mean = 3.0
norm_std = 1.0
print("norm_mean: " .. norm_mean .. ", norm_std: " .. norm_std)
x_train_n = (x_train - norm_mean) / norm_std

-- Normalize the test inputs according to the training data normalization
-- parameters.
x_test_n = (x_test - norm_mean) / norm_std

for i=1,1 do
  x_train_n:select(3,i):cmul(mask_train)
  x_test_n:select(3,i):cmul(mask_test)
end

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
    local inputs = x:narrow(1,start,batchSize):reshape(batchSize,maxLength,1)
    local batchLengths = lengths:narrow(1,start,batchSize)

    local targets = torch.reshape(y:narrow(1,start,batchSize), batchSize, maxLength, 1)
    -- Provide a matrix that masks unused portions of the outputs. This will be
    -- ones over the length of the sequence and zeros afterwards. 
    local validSequence = torch.zeros(batchSize,maxLength)
    for b=1,batchSize do
      local len = batchLengths[b]
      validSequence[b]:narrow(1,1,len):fill(1)
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
  
  function trainer:train(dataset)
    local iteration = 1
    local currentLearningRate = self.learningRate
    local module = self.module
    local criterion = self.criterion
  
    local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
  
    local par = module.par
    local gradPar = module.gradPar
    --par:uniform(-0.02, 0.02)

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

-- chainIn will be a table like {inputSeq,lengths,validSeq}. We'll use validSeq
-- this to constrain predictions to the portions of the sequence we care about in
-- each batch member, given they will have different lengths.
chainIn = nn.Identity()()
inputSeq = nn.SelectTable(1)(chainIn)
batchLengths = nn.SelectTable(2)(chainIn)
validSeq = nn.SelectTable(3)(chainIn)
reversedInputSeq = lstm.ReverseSequence(1)({inputSeq,batchLengths})
chainModForward = lstm.MemoryChainDirect(1, {params.hidden}, maxLength)
chainOutForward = chainModForward({inputSeq, batchLengths})
-- The backward chain will take output from the forward chain and the input
chainOutForwardReversed = lstm.ReverseSequence(1)({chainOutForward,batchLengths})
backwardWithInput = nn.JoinTable(3)({reversedInputSeq, chainOutForwardReversed})
chainModBackward = lstm.MemoryChainDirect(1+params.hidden, {params.hidden}, maxLength)
chainOutBackward = chainModBackward({backwardWithInput, batchLengths})
chainOut = lstm.ReverseSequence(1)({chainOutBackward,batchLengths})

-- In order to feed these through the linear map for prediction, we have to
-- reshape so that we have a 2D tensor instead of 3D. Later we reshape it
-- back.
chainOutReshaped = nn.Reshape(params.batch*maxLength, params.hidden)(chainOut)
predicted = nn.Linear(params.hidden,1)(chainOutReshaped)
predictedReshaped = nn.Reshape(params.batch, maxLength, 1)(predicted)
predictedConstrained = nn.CMulTable()({predictedReshaped,validSeq})
net = nn.gModule({chainIn},{predictedConstrained})

-- Need to reenable sharing after getParameters(), which broke my sharing.
net.par, net.gradPar = net:getParameters()
chainModForward:setupSharing()
chainModBackward:setupSharing()

-- Use least-squares loss function and SGD.
criterion = nn.MSECriterion()
--criterion.sizeAverage = false

--net.par:copy(torch.load("par"))
net.par:uniform(-0.02, 0.02)
torch.manualSeed(params.seed)

for i=1,2 do
  print("## ITERATION " .. i)
  print("input[1]:")
  print(trainingDataset[i][1][1]:view(-1,maxLength))
  print("input[2]:")
  print(trainingDataset[i][1][2])
  print("input[3]:")
  print(trainingDataset[i][1][3])
  print("targets:")
  print(trainingDataset[i][2]:view(-1,maxLength))
  net:zeroGradParameters()
  net:forward(trainingDataset[i][1])
  print("net.output:")
  print(net.output:view(-1,maxLength))
  criterion:forward(net.output, trainingDataset[i][2])
  print("error: " .. criterion.output)
  print("sse: " .. (net.output - trainingDataset[i][2]):pow(2):sum())
  criterion:backward(net.output, trainingDataset[i][2])
  net:backward(trainingDataset[i][1], criterion.gradInput)
  print("net.gradInput:")
  print(net.gradInput[1]:view(-1,maxLength))
  print("gradPar:sum(): " .. net.gradPar:sum())
end
--os.exit(0)

--if params.mode == 'train' then
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
  os.exit(0)

  -- Save the trained model
  torch.save(params.trained, {net=net})

  -- Output predictions along a grid so we can see how well it learned the function. We'll
  -- generate inputs without noise so we can see how well it does in the absence of noise,
  -- which will give us a sense of whether it's learned the true underlying function.
  grid_size = math.ceil(200/params.batch)*params.batch
  target_grid_orig = torch.linspace(0, toy.max_target, grid_size):view(grid_size,1)
  inputs_grid, target_grid = toy.direct_target_to_inputs(target_grid_orig, 0, maxLength)
  inputs_grid_n = (inputs_grid - norm_mean) / norm_std
  -- use length 3 sequences
  target_grid:narrow(2,4, maxLength-3):fill(0)
  inputs_grid_n:narrow(2,4, maxLength-3):fill(0)
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
  pred_d = pldata.new(allPredictions:narrow(2,1,3):reshape(grid_size,3):totable())
  pred_d:write(params.grid)

--elseif params.mode == 'check' then
--
--  -- Check gradients for the first training example
--  example = trainingDataset[1]
--  exampleLengths = example[1][2]:nonzero():select(2,2)
--  -- This method returns a vector containing L ones with the rest zeros.
--  local mapRow = function(L)
--    v = torch.zeros(4)
--    v:narrow(1,1,L):fill(1)
--    return v:view(1,-1)
--  end
--  -- We use mapRow to make a mask matrix so we can zero out inputs that
--  -- are not really part of each example.
--  mask = nn.JoinTable(1):forward(tablex.map(mapRow, exampleLengths:totable()))
--  local get1 = function(z) return z[1] end
--  local err = check.checkInputsGrad(net, criterion, example, example[1][1], mask, get1)
--  print("error in estimate of inputs Jacobian: " .. err)
--  err = check.checkParametersGrad(net, criterion, example, net.par, net.gradPar, get1)
--  print("error in estimate of parameters Jacobian: " .. err)
--
--else
--  error("invalid mode " .. params.mode)
--end
-- END
