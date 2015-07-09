-- Here I try modeling two criteria at once, one real and one a discrete version
-- of that. The main motivation is to try predicting multiple things at once.

require 'optim'
plutils = require 'pl.utils'
tablex = require 'pl.tablex'
lstm = require 'lstm'

inputs = {}
total_count = 0
train_points = 0
lines = plutils.readlines('noisy_inputs.txt')
for i,line in pairs(lines) do
  local vals = plutils.split(line, ',')
  inputs[i] = torch.Tensor(tablex.map(tonumber, vals))
  total_count = total_count + 1
end

true_inputs = {}
total_count = 0
lines = plutils.readlines('true_inputs.txt')
for i,line in pairs(lines) do
  local vals = plutils.split(line, ',')
  true_inputs[i] = torch.Tensor(tablex.map(tonumber, vals))
  total_count = total_count + 1
end

-- For this tests, I model the same thing but two ways. On the real line using
-- MSECriterion and as 10 discrete classes dividing the real line and using
-- CrossEntropyCriterion.
num_classes = 10
outputs = tablex.map(tonumber, plutils.readlines('outputs.txt'))
outputs = torch.Tensor(outputs)
output_range = outputs:max() - outputs:min() + 1e-08
output_classes = (outputs - outputs:min() + 1e-09):div(output_range/num_classes):ceil()
output_tuples = torch.cat(outputs, output_classes, 2)

-- Decide on the train/test split
test_frac = 0.3
m = torch.floor(total_count * test_frac)
n = total_count - m

-- Split the data into test and train
x_train = {}
x_test = {}
for i,input in pairs(inputs) do
  if i <= n then
    x_train[i] = input
    train_points = train_points + input:size(1)
  else
    x_test[i-n] = input
  end
end

y_train = output_tuples:narrow(1, 1, n)
y_test = output_tuples:narrow(1, n+1, m)

-- Normalize the inputs
norm_mean =
  torch.Tensor(tablex.map(function(i) return i:sum() end, x_train)):sum() / train_points
norm_var = torch.Tensor(tablex.map(
  function(i) return (i - norm_mean):pow(2):sum() end,
  x_train)):sum() / train_points
norm_std = math.sqrt(norm_var)

x_train_n = tablex.map(function(z) return (z - norm_mean) / norm_std end, x_train)
x_test_n = tablex.map(function(z) return (z - norm_mean) / norm_std end, x_test)
x_true_n = tablex.map(function(z) return (z - norm_mean) / norm_std end, true_inputs)

function buildDataset(x, y)
  local dataset = {}
  local n = #x
  function dataset:size()
    return n
  end
  for i=1,dataset:size() do 
    dataset[i] = {x[i]:view(-1,1), y:narrow(1,i,1)}
  end
  return dataset
end

train_dataset = buildDataset(x_train_n, y_train)
test_dataset = buildDataset(x_test_n, y_test)
true_dataset = buildDataset(x_true_n, output_tuples)

n_h = 20
max_len = 4

net = {}
net.classes = num_classes
-- LSTM
x = nn.Identity()()
net.chain = lstm.MemoryChain(1, n_h, max_len)
lastHidden = nn.SelectTable(1)(net.chain(x))
net.chainMod = nn.gModule({x}, {lastHidden})

-- Prediction
chainOut = nn.Identity()()
predictReal = nn.Linear(n_h, 1)(chainOut)
predictClass = nn.LogSoftMax()(nn.Linear(n_h, num_classes)(chainOut))
predictions = nn.Identity()({predictReal, predictClass})
-- Takes the output from the LSTM chain and does multi-output prediction. Output
-- will be {1x1, 1xC}, where C is the number of classes.
net.predict = nn.gModule({chainOut}, {predictions})
-- Criterion
rC = nn.MSECriterion()
cC = nn.ClassNLLCriterion()
net.criterion = nn.ParallelCriterion():add(rC):add(cC)
-- Get access to net's parameters
net.predict.par_vec, net.predict.grad_par_vec = net.predict:getParameters()

trainer = {}
trainer.maxIteration = 25
trainer.learningRate = 0.01
function trainer:hookIteration(iter, err)
  print("# current error = " .. err)
end

function trainer:train(net, dataset)
  local iteration = 1
  local lr = self.learningRate
  local shuffledIndices = torch.randperm(dataset:size(), 'torch.LongTensor')
  local chain = net.chain
  local chainMod = net.chainMod
  local predict = net.predict
  local criterion = net.criterion
  while true do
    local currentError = 0
    for t = 1,dataset:size() do
      local example = dataset[shuffledIndices[t]]

      net:forward(unpack(example))
      net:backward(unpack(example))

      -- Update parameters
      chain.lstm_par_vec:add(chain.lstm_grad_par_vec:mul(-lr))
      predict.par_vec:add(predict.grad_par_vec:mul(-lr))
      currentError = currentError + criterion.output
    end
    currentError = currentError / dataset:size()

    if self.hookIteration then
      self.hookIteration(self, iteration, currentError)
    end

    iteration = iteration + 1
    if self.maxIteration > 0 and iteration > self.maxIteration then
      print("# Max iteration reached; training error = " .. currentError)
      break
    end
  end
end

function net:forward(input, targets)
  targets = self:splitTargets(targets)
  self.chainMod:forward(input)
  self.predict:forward(self.chainMod.output)
  return self.criterion:forward(self.predict.output, targets)
end

function net:splitTargets(targets)
  local targetReal = targets:select(2,1):view(-1,1)
  local targetClass = targets:select(2,2)
  return {targetReal, targetClass}
end

function net:backward(input, targets)
  targets = self:splitTargets(targets)
  self.chain:zeroGradParameters()
  self.predict:zeroGradParameters()
  self.criterion:backward(self.predict.output, targets)
  self.predict:backward(self.chain.output, self.criterion.gradInput)
  self.chainMod:backward(input, self.predict.gradInput)
end

function net:predictionError(dataset)
  local err = 0
  for i=1, dataset:size() do
    self:forward(unpack(dataset[i]))
    err = err + self.criterion.output
  end
  return err / dataset:size()
end

function net:classify(dataset)
  local predicted_classes = torch.Tensor(dataset:size())
  for i=1, dataset:size() do
    local input, target = unpack(dataset[i])
    net:forward(input, target)
    local odds = net.predict.output[2]
    _, cls = torch.max(nn.LogSoftMax():forward(odds), 2)
    predicted_classes[i] = cls
  end
  return predicted_classes
end

function net:confusion(dataset)
  if self.classes == nil then
    error('not a classification net')
  end
  confusion = optim.ConfusionMatrix(torch.range(1,self.classes):totable())
  local predictions = self:classify(dataset)
  for i=1, dataset:size() do
    local predicted = predictions[i]
    local actual = dataset[i][2][{1,2}]
    confusion:add(predicted, actual)
  end
  return confusion
end

-- Train the model
trainer:train(net, train_dataset)

-- See how it did
net:confusion(test)



-- END

