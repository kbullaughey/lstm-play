-- The methods provided by this script allow one to check the gradients
-- of a nerual net are properly computed.
local check = {dx=0.001}

-- Returns a linearized view of tensor that can use get() and set() methods
-- to access only the masked elements.
check.maskedTensor = function(tensor, mask)
  local z = {}
  z.view = tensor:view(-1)
  local fullLength = z.view:size(1)
  if mask then
    z.mask = mask:view(-1)
  else
    z.mask = torch.ones(fullLength)
  end
  z.nonzero = torch.nonzero(z.mask):view(-1)
  z.len = z.nonzero:size(1)
  function z:get(i)
    return self.view[self.nonzero[i]]
  end
  function z:set(i, value)
    self.view[self.nonzero[i]] = value
  end
  -- Return a copy containing all unmasked values in a vector
  function z:extract()
    local r = torch.Tensor(self.len)
    for i=1,self.len do
      r[i] = self:get(i)
    end
    return r
  end
  return z
end

-- Abstraction that can handle computing finite-difference gradient for either
-- inputs or parameters. view is a linearized (potentially masked) vector of
-- interest. view should be something returned from check.maskedTensor.
check.computeFD = function(view, net, criterion, example)
  local inputs, targets = unpack(example)
  local predictions = net:forward(inputs)
  local f0 = criterion:forward(predictions, targets)
  -- Estimate the partial derivatives of outputs with respect to inputs using
  -- finite difference approximation.
  local fd = torch.zeros(view.len)
  for i=1,view.len do
    local originalValue = view:get(i)
    view:set(i, originalValue + check.dx)
    predictions = net:forward(inputs)
    local perturbed = criterion:forward(predictions, targets)
    fd[i] = (perturbed - f0) / check.dx
    view:set(i, originalValue)
  end
  return fd
end

-- example is assumed to be a two-item table with the first item giving
-- the content passed to net:forward() and the second item is the second argument
-- to criterion. If not all inputs are actual inputs (say to to varying sequence
-- lengths) then a mask can be provided. actualInputs are the inputs to be
-- perturbed and must match mask in length. It's okay if example[1] contains
-- additional stuff besides what's in actualInputs. This method assumes example
-- is a mini-batch with fist dimension iterating over examples in the batch.
check.inputsGradFD = function(net, criterion, example, actualInputs, mask)
  local inputView = check.maskedTensor(actualInputs, mask)
  return check.computeFD(inputView, net, criterion, example)
end

-- Run forward and backward.
check.forwardBackward = function(net, criterion, example)
  local inputs, targets = unpack(example)
  net:zeroGradParameters()
  net:forward(inputs)
  criterion:forward(net.output, targets)
  criterion:backward(net.output, targets)
  net:backward(inputs, criterion.gradInput)
end

check.inputsGrad = function(net, criterion, example)
  check.forwardBackward(net, criterion, example)
  return net.gradInput
end

check.checkInputsGrad = function(net, criterion, example, actualInputs, mask)
  local grad = check.maskedTensor(check.inputsGrad(net, criterion, example), mask):extract()
  local fd = check.inputsGradFD(net, criterion, example, actualInputs, mask)
  return torch.sqrt((fd - grad):pow(2):sum())
end

-- In order to allow flexibility on where the gradient parameters are stored, I
-- require them to be passed to the function.
check.parametersGrad = function(net, criterion, example, gradParams)
  check.forwardBackward(net, criterion, example)
  return gradParams
end

check.parametersGradFD = function(net, criterion, example, par)
  local parView = check.maskedTensor(par)
  return check.computeFD(parView, net, criterion, example)
end

-- In order to allow flexibility on where the parameters are stored, I require
-- both the parameter and gradient parameters to be passed to the function.
check.checkParametersGrad = function(net, criterion, example, params, gradParams)
  local grad = check.parametersGrad(net, criterion, example, gradParams):view(-1)
  local fd = check.parametersGradFD(net, criterion, example, params)
  return torch.sqrt((fd - grad):pow(2):sum())
end

return check
