nn = require 'nn'

toy = {}

-- This is our arbitrary wavy function. We will try and learn the inverse of this.
-- Expects an tensor or a scalar number.
-- 2 + 0.5*sin(x/2 - 1) + sin(x) + sin(x*2 + 2) + sin(x/4 + 1)
toy.f = function(x) 
  -- Convert to tensor if not
  if not torch.isTensor(x) then
    x = torch.Tensor(1):fill(x)
  end
  -- Weighted average for four arbitrary sine waves.
  return (
    torch.sin(x/2 - 1):mul(0.5) +
    torch.sin(x) +
    torch.sin(x*2 + 2) +
    torch.sin(x/4 + 1) + 2)
end

-- Our targets will be in (0,max_target)
toy.max_target = 10

-- The inputs are derived from the target by evaluating the function above at
-- time lags of 0, 1, and 2. i.e. f(target), f(target-1), f(target-2). Noise
-- is then added by sampling N(0,sd)
toy.target_to_inputs = function(target, sd, len)
  -- Specify noise standard deviation if this optional parameter is not given
  sd = sd or 0.2
  len = len or 3
  local seq = {}
  for i=1,len do
    seq[i] = target-i+1
  end
  local inputs = toy.f(nn.JoinTable(2):forward(seq))
  -- Add some noise to the inputs
  local n = target:size(1)
  return inputs + torch.randn(n,len):mul(sd)
end

return toy

-- END
