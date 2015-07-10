-- enable profiling
--jitp = require('jit.p')
--jitp.start("Fl1i1")

torch.setdefaulttensortype('torch.FloatTensor')

use_cuda = true

require 'nn'
if use_cuda then
  use_cda, cunn = require('fbcunn')
end
if use_cuda then
  deviceParams = cutorch.getDeviceProperties(1)
  Tensor = torch.CudaTensor
else
  Tensor = torch.Tensor
end

local n = 200


k = 1000
nodes = {}
for i=1,k do
  local linearMap = nn.Linear(n,n)
  if use_cuda then
    linearMap = linearMap:cuda()
  end
  local par = linearMap:getParameters()
  par:uniform(-1,1)
  nodes[i] = linearMap
end
start_time = os.time()

dataset = {}
m = 100000
for i=1,m do
  local x = Tensor(2,n):uniform()
  local y = Tensor(2,n):uniform()
  local j = (i-1) % k + 1
  if j == 1 and use_cuda then
    cutorch.synchronize()
  end
  local map = nodes[j]
  map:zeroGradParameters()
  map:forward(x)
  map:backward(x,y)
end
print(os.time() - start_time)


--jitp.stop()
-- END
