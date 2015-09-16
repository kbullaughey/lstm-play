nn = require 'nn'

toy = {}

-- This is our arbitrary wavy function. We will try and learn the inverse of this.
-- Expects an tensor or a scalar number.
toy.f = function(x) 
  -- Convert to tensor if not
  if not torch.isTensor(x) then
    x = torch.Tensor(x)
  end
  -- Weighted average for four arbitrary sine waves.
  return (
    torch.sin(x + 1):mul(4) +
    torch.sin(x*2 + 1) +
    torch.sin(x*4):mul(2) +
    torch.sin(x/2):mul(3))
end

-- Our targets will be in (0,max_target)
toy.max_target = 10
toy.num_inputs = 3

-- The inputs are derived from the target by evaluating the function above at
-- time lags of 0, 1, and 2. i.e. f(target), f(target-1), f(target-2). Noise
-- is then added by sampling N(0,sd)
toy.target_to_inputs = function(target, sd)
  -- Specify noise standard deviation if this optional parameter is not given
  sd = sd or 0.2
  local inputs = toy.f(nn.JoinTable(2):forward({target, target-1, target-2}))
  -- Add some noise to the inputs
  local n = target:size(1)
  return inputs + torch.randn(n,3):mul(sd)
end

return toy

-- END
