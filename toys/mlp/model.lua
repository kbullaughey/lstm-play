-- Here I model a fixed number of time steps, three, using a non-recurrent
-- MLP. 

require 'csvigo'
require 'nn'
tablex = require 'pl.tablex'

-- Read the file in from csv. Four columns x1, x2, x3, y
d = csvigo.load("data.csv")

-- Make tensors from data table
num_cols = 3
local columns = {}
for i=1,num_cols do
  local key = "x" .. i
  columns[i] = tablex.map(tonumber, d[key])
end
x = torch.Tensor(columns):t()
y = torch.Tensor(tablex.map(tonumber, d.y))

total_count = x:size(1)

-- Decide on the train/test split
test_frac = 0.3
m = torch.floor(total_count * test_frac)
n = total_count - m

-- Split the data into test and train
x_train = x:narrow(1, 1, n)
y_train = y:narrow(1, 1, n)
x_test = x:narrow(1, n+1, m)
y_test = y:narrow(1, n+1, m)

-- Normalize the inputs
norm_mean = x_train:mean()
norm_std = x_train:std()
x_train_n = (x_train - norm_mean) / norm_std
x_test_n = (x_test - norm_mean) / norm_std

-- Set up a data structure
dataset={};
function dataset:size()
  return n
end
for i=1,dataset:size() do 
  dataset[i] = {x_train_n:narrow(1,i,1), y_train:narrow(1,i,1)}
end

-- Set up the neural net
net_size = 40
mlp = nn.Sequential()
mlp:add(nn.Linear(num_cols, net_size))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(net_size, 1))

-- Use least-squares loss function and SGD.
criterion = nn.MSECriterion()

-- Train the model
trainer = nn.StochasticGradient(mlp, criterion)
trainer.maxIteration = 40
trainer.learningRate = 0.01
function trainer:hookIteration(iter)
  print("# test error = " .. criterion:forward(mlp:forward(x_test_n), y_test))
end
trainer:train(dataset)

-- See how predictions to on noisless data.
grid2 = csvigo.load("grid2.csv")
columns = {}
for i=1,num_cols do
  local key = "x" .. i
  columns[i] = tablex.map(tonumber, grid2[key])
end
x_grid = torch.Tensor(columns):t()
x_grid_n = (x_grid - norm_mean) / norm_std

y_grid = mlp:forward(x_grid_n)
csvigo.save{path='gridPredict2.csv', data=y_grid:totable()}

-- END
