#!/usr/bin/env th

toy = require '../toy/toy'
require 'lstm'

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
cmd:option('-trained','trained_model-1_layer-fixed_rock.t7','file name for saved trained model')
cmd:option('-grid','grid_predictions-1_layer-fixed_rock.csv','file name for saved grid predictions')
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
maxLength = 4

-- Separate data into inputs (x) and targets (y)
x = torch.Tensor(N,4)
x:narrow(2, 1, numInput):copy(d:narrow(2, 1, numInput))
lengths = torch.Tensor(N):fill(3)
y = d:narrow(2, numInput+1, 1)

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

print("norm_mean: " .. norm_mean)
print("norm_std: " .. norm_std)
print("x_train_n:sum(): " .. x_train_n:sum())
print("x_test_n:sum(): " .. x_test_n:sum())

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
    local targets = torch.reshape(y:narrow(1,start,batchSize), batchSize, 1, 1)
    -- Encode lengths using a one-hot per row strategy
    local batchLengths = torch.zeros(batchSize,maxLength)
    for b=1,batchSize do
      batchLengths[b][lengths:narrow(1,start,batchSize)[b]] = 1
    end
    -- Add a zero matrix to every example for the initial h state
    dataset[i] = {{inputs,batchLengths}, targets}
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

chainIn = nn.Identity()()
chainMod = lstm.MemoryChain(1, {params.hidden}, maxLength)
chainOut = chainMod(chainIn)
predictedMod = nn.Linear(params.hidden,1)
predicted = predictedMod(chainOut)
net = nn.gModule({chainIn},{predicted})

-- Need to reenable sharing after getParameters(), which broke my sharing.
net.par, net.gradPar = net:getParameters()
chainMod:setupSharing()

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

-- END
