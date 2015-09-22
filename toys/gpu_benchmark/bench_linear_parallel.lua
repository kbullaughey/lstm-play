#!/usr/bin/env th

socket = require 'socket'

-- enable profiling
--jitp = require('jit.p')
--jitp.start("Fl1i1")

torch.setdefaulttensortype('torch.FloatTensor')

cmd = torch.CmdLine()
cmd:text()
cmd:text('Benchmarking using a GPU')
cmd:text()
cmd:text('Options')
cmd:option('-seed',os.time(),'initial random seed (defaults to current time)')
cmd:option('-M',1000,'Linear map will be MxM')
cmd:option('-gpu',false,'uses GPU when flag present')
cmd:option('-reps',10000,'replicates')
cmd:option('-maps',10,'Number of linear maps per synchronization')
cmd:text()

-- parse input params
params = cmd:parse(arg)

use_cuda = params.gpu
local M = params.M
local K = params.maps
local reps = params.reps

require 'nn'
if use_cuda then
  cutorch = require 'cutorch'
  deviceParams = cutorch.getDeviceProperties(1)
  Tensor = torch.CudaTensor
else
  Tensor = torch.Tensor
end

print("# setting up.")
nodes = {}
for i=1,K do
  local linearMap = nn.Linear(M,M)
  if use_cuda then
    linearMap = linearMap:cuda()
  end
  local par = linearMap:getParameters()
  par:uniform(-1,1)
  nodes[i] = linearMap
end

print("# running.")
start_time = socket.gettime()
for i=1,reps do
  local x = Tensor(2,M):uniform()
  local y = Tensor(2,M):uniform()
  local j = (i-1) % K + 1
  if j == 1 and use_cuda then
    cutorch.synchronize()
  end
  local map = nodes[j]
  map:zeroGradParameters()
  map:forward(x)
  map:backward(x,y)
end
elapsedTime = socket.gettime() - start_time
print("# done.")
print("# M,maps,reps,mode,elapsed_time")
local gpuString
if params.gpu then
  gpuString = 'gpu'
else
  gpuString = 'cpu'
end
print(params.M .. ',' .. params.maps .. ',' .. params.reps .. ',' ..
  gpuString .. ',' .. elapsedTime)



--jitp.stop()
-- END
