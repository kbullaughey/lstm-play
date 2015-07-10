-- enable profiling
--jitp = require('jit.p')
--jitp.start("Fl1i1")

torch.setdefaulttensortype('torch.FloatTensor')

use_cuda = false

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

local n = 4000
local linearMap = nn.Linear(n,n)
if use_cuda then
  linearMap = linearMap:cuda()
end

local par = linearMap:getParameters()
par:uniform(-1,1)

for i=1,1000 do
  local x = Tensor(2,n):uniform()
  local y = Tensor(2,n):uniform()
  linearMap:zeroGradParameters()
  linearMap:forward(x)
  linearMap:backward(x,y)
  --cutorch.synchronize()
end

--jitp.stop()
-- END
