local BasicNet, parent = torch.class('lstm.BasicNet', 'nn.Module')

-- This network is just a shell.
--
-- The user must define three properties:
--    chain - a lstm.MemoryChain instance
--    predict
--    criterion
--
-- Such that they can be chained together in forward and backward prediction.
--
-- After creating chain, predict, and criterion, one must call:
--
--    net:finalize()
--
-- Which handles getting access to parameters and sharing.
--
-- In order to support dropoff, methods must be provided to turn droppoff on
-- and off. These are labeled `turnDropoffOn` and `turnDropoffOff`.
--
function BasicNet:__init()
  parent.__init(self)
end

function BasicNet:type(type)
  self.gradInput = {}
  return parent.type(self, type)
end

function BasicNet:parameters()
  local chain_par, chain_grad_par = self.chain:parameters()
  local predict_par, predict_grad_par = self.predict:parameters()
  local par = {}
  local grad_par = {}
  local j = 1
  for i=1, #chain_par do
    par[j] = chain_par[i]
    grad_par[j] = chain_grad_par[i]
    j = j + 1
  end
  for i=1, #predict_par do
    par[j] = predict_par[i]
    grad_par[j] = predict_grad_par[i]
    j = j + 1
  end
  return par, grad_par
end

-- Expects a tuple of the form ((inputs,lengths),targets). Assumes batch mode, so
-- if mini-batches are wanted, make sure the first dimension of these tensors
-- has size 1.
function BasicNet:updateOutput(nested_tuple)
  self:ensureFinalized()
  local input_and_lengths, targets = unpack(nested_tuple)
  local input, lengths = unpack(input_and_lengths)
  self.chain:forward({input, lengths})
  self.predict:forward(self.chain.output)
  self.output = self.criterion:forward(self.predict.output, targets)
  return self.output
end

function BasicNet:reset(radius)
  radius = radius or 0.1
  self:zeroGradParameters()
  self.par_vec:uniform(-radius, radius)
end

function BasicNet:updateGradInput(input_and_lengths, targets)
  self:ensureFinalized()
  self:zeroGradParameters()
  self.criterion:backward(self.predict.output, targets)
  self.predict:backward(self.chain.output, self.criterion.gradInput)
  self.chain:backward(input_and_lengths, self.predict.gradInput)
  self.gradInput = self.chain.gradInput
  lstm.sync()
  return self.gradInput
end

function BasicNet:ensureFinalized()
  if not self.finalized then
    error("Net has not been finalized")
  end
end

function BasicNet:finalize()
  if self.finalized then
    error("Net has already been finalized")
  end
  self.par_vec, self.grad_par_vec = self:getParameters()
  self.chain:share()
  self.finalized = true
end

-- This must be implemented to support dropoff.
function BasicNet:turnDropoffOff()
end

-- This must be implemented to support dropoff.
function BasicNet:turnDropoffOn()
end

function BasicNet:training()
  self.predict:training()
end

function BasicNet:evaluate()
  self.predict:evaluate()
end

function BasicNet:grad_wrt_inputs_finite_difference(example, dx)
  local input_and_lengths, target = unpack(example)
  local input, lengths = unpack(input_and_lengths)
  local batch_size = input:size(1)
  local len = input:size(2)
  local num_features = input:size(3)
  self:forward(example)
  local f0 = self.criterion.output

  -- Estimate the partial derivatives of outputs with respect to inputs using
  -- finite difference approximation.
  local fd = lstm.Tensor()(batch_size, len, num_features):zero()
  for b=1, batch_size do
    for i=1, len do
      for j=1, num_features do
        local original_value = input[b][i][j]
        input[b][i][j] = original_value + dx
        self:forward({{input, lengths}, target})
        local perturbed = self.criterion.output
        local grad_est = (perturbed - f0) / dx
        fd[b][i][j] = grad_est
        input[b][i][j] = original_value
      end
    end
  end
  return fd
end

-- Compute the partial derivatives of outputs with respect to inputs
function BasicNet:grad_wrt_inputs(example)
  local input_and_lengths, target = unpack(example)
  self:zeroGradParameters()
  self:forward(example)
  self:backward(input_and_lengths, target)
  return self.gradInput
end

-- Return just the components of the gradient wrt inputs that correspond to
-- parts of the actual sequences of each batch memory, i.e., throw away
-- the filler for shorter sequences.
function BasicNet:filter_grad_wrt_inputs(grad, lengths)
  local pieces = {}
  local batch_size = lengths:size(1)
  for b=1, batch_size do
    local this_length = lengths[b]
    pieces[b] = grad:sub(b, b, 1, this_length):view(-1)
  end
  return lstm.localize(nn.JoinTable(1)):forward(pieces)
end

function BasicNet:check_grad_wrt_inputs(example)
  local input_and_lengths, target = unpack(example)
  local input, lengths = unpack(input_and_lengths)
  self:turnDropoffOff()
  local fd = self:grad_wrt_inputs_finite_difference(example, 0.001)
  local grad = self:grad_wrt_inputs(example)
  -- Strip out the filler that is added because some sequences in the batch are
  -- shorter than others.
  fd = self:filter_grad_wrt_inputs(fd, lengths)
  grad = self:filter_grad_wrt_inputs(grad, lengths)
  self:turnDropoffOn()
  return (fd - grad):pow(2):sum()
end

function BasicNet:grad_wrt_parameters_finite_difference(example, dx)
  local num_par = self.par_vec:size(1)
  self:forward(example)
  local f0 = self.criterion.output
  -- Perturb each of the parameters individually
  -- Compute what we expect the finite-difference approximation to the gradient
  local fd = lstm.Tensor()(num_par, 1):zero()
  for i=1,num_par do
    local original_value = self.par_vec[i]
    self.par_vec[i] = original_value + dx
    self:forward(example)
    local perturbed = self.criterion.output
    local grad_est = (perturbed - f0) / dx
    fd[i]:fill(grad_est)
    self.par_vec[i] = original_value
  end
  return fd
end

function BasicNet:grad_wrt_parameters(example)
  num_par = self.par_vec:size(1)
  self:zeroGradParameters()
  -- Compute the partial derivatives of outputs with respect to parameters 
  self:forward(example)
  local input, target = unpack(example)
  self:backward(input, target)
  return self.grad_par_vec:clone()
end

function BasicNet:check_grad_wrt_parameters(example)
  self:turnDropoffOff()
  local f1 = self:grad_wrt_parameters_finite_difference(example, 0.001)
  local grad = self:grad_wrt_parameters(example)
  self:turnDropoffOn()
  return (f1 - grad):pow(2):sum()
end


return BasicNet

-- END
