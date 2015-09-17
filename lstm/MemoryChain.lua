local MemoryChain, parent = torch.class('lstm.MemoryChain', 'nn.Module')

function MemoryChain:__init(inputSize, hiddenSize, maxLength)
  print("MemoryChain("..inputSize..','..hiddenSize..','..maxLength..')')
  parent.__init(self)
  self.inputSize = inputSize
  self.hiddenSize = hiddenSize
  self.maxLength = maxLength
  self.gradInput = nil
  self.lstms = {}
  -- Later we will cache the batch size and length of the input during the
  -- forward so we don't bother recomputing these during the backward pass.
  self.batchSize = nil
  self.len = nil

  print("Creating MemoryChain")
  -- make enough lstm cells for the longest sequence
  for i=1,maxLength do
    self.lstms[i] = lstm.MemoryCell(inputSize, hiddenSize)
  end
  -- We will use the firt cell as the storage for shared parameters, but
  -- we can't share them until later, after the whole network is created 
  -- because getParameters will point the tensors to new storage.
  self.lstm_params, self.lstm_grad_params = self.lstms[1]:parameters()
end

-- share parameters among all memory cells
function MemoryChain:share()
  -- make all other parameter tensors reference that memory.
  for i=2,self.maxLength do
    local cell_params, cell_grad_params = self.lstms[i]:parameters()
    for k=1, #cell_params do
      cell_params[k]:set(self.lstm_params[k])
      cell_grad_params[k]:set(self.lstm_grad_params[k])
    end
  end
end

function MemoryChain:reset(radius)
  local par = self:parameters()
  for i=1, #par do
    par[i]:uniform(-radius, radius)
  end
end

function MemoryChain:parameters()
  return self.lstm_params, self.lstm_grad_params
end

-- Receives a table containing two Tensors: input and a vector of lengths, as not all
-- sequences will span the full length dimension of the tensor.
-- If input is 3D then the first dimension is batch, otherwise the first dim
-- is the sequence. Last dimension is features.
function MemoryChain:updateOutput(tuple)
  local input, lengths = unpack(tuple)
  if input:dim() ~= 3 then
    error("expecting a 3D input")
  end
  self.batchSize = input:size(1)
  self.len = input:size(2)
  -- The first memory cell will receive zeros.
  local h = lstm.Tensor()(self.batchSize, self.hiddenSize):zero()
  local c = lstm.Tensor()(self.batchSize, self.hiddenSize):zero()
  -- Store inputs of each memory cell so we have these for the backwards pass.
  self.hidden_states = {[0] = h}
  self.memories = {[0] = c}
  -- Iterate over memory cells feeding each successive tuple (h,c) into the next
  -- LSTM memory cell.
  for i=1,self.len do
    local x = input:select(2, i)
    self.lstms[i]:forward({x, h, c})
    h, c = unpack(self.lstms[i].output)
    self.hidden_states[i] = h
    self.memories[i] = c
  end
  -- Copy the terminal output for each batch member into the output tensor.
  self.output = lstm.Tensor()(self.batchSize, self.hiddenSize)
  for b=1, self.batchSize do
    h = self.lstms[lengths[b]].output[1]
    self.output[b]:copy(h[b])
  end
  return self.output
end

-- upstreamGradOutput will be a BxH matrix where B is batch size and H is hidden
-- state size. Each row will correspond to the gradient of the object function
-- wrt outputs from the LSTM memory cell corresponding to the sequence terminus.
-- However, this isn't necessarily the last memory cell in the `lstms` array
-- because sequences are different lengths.
function MemoryChain:updateGradInput(tuple, upstreamGradOutput)
  local input, lengths = unpack(tuple)
  local h,c
  if input:dim() ~= 3 then
    error("MemoryChain:updageGradInput is expecting a 3D input tensor")
  end
  self.gradInput = lstm.Tensor()(self.batchSize, self.len, self.inputSize):zero()
  -- Because each batch member has a sequence of a different length less than
  -- or equal to self.len, we need to have some way to propagate errors starting
  -- at the correct level. 
  --
  -- I build a binary matrices of size BxL. This matrix will be used to
  -- determine from where error terms are propagating back from. The matrix,
  -- terminal, has a one at the terminal column in the sequence. 
  local terminal = lstm.Tensor()(self.batchSize, self.len):zero()
  for b=1,self.batchSize do
    local l = lengths[b]
    terminal[b][l] = 1
  end
  -- Since we only use the hidden state for prediction, the gradient wrt to the
  -- memory is zero. And since beyond the end of the sequence the extra memory
  -- cells can't affect the objective function, the gradient wrt to the extra
  -- hidden states is also zero.
  local gradOutput = {
    lstm.Tensor()(self.batchSize, self.hiddenSize):zero(),
    lstm.Tensor()(self.batchSize, self.hiddenSize):zero()
  }
  for i=self.len,1,-1 do
    -- Replicate our mask of which batch members receive errors from the
    -- upstream gradient at this time step. (replicated hiddenSize times)
    local terminalColumn = terminal:select(2, i):contiguous():view(-1,1)
    local upstream = torch.mm(terminalColumn, lstm.localize(torch.ones(1,self.hiddenSize)))
    local x = input:select(2,i)
    h = self.hidden_states[i-1]
    c = self.memories[i-1]
    gradOutput[1]:add(upstream:cmul(upstreamGradOutput))
    self.lstms[i]:backward({x,h,c}, gradOutput)
    gradOutput = self.lstms[i].gradInput
    -- Only h and c propagate back to the next cell, the gradient wrt x gets stored
    -- in gradInput.
    self.gradInput:select(2, i):copy(gradOutput[1])
    gradOutput = {gradOutput[2], gradOutput[3]}
  end
  return self.gradInput
end

-- This happens automatically when calling backward on the individual memory
-- cells in updateGradInput. Not sure what to do about the scale parameter.
function MemoryChain:accGradParameters(input, gradOutput, scale)
end

function MemoryChain:type(type)
  self.gradInput = {}
  return parent.type(self, type)
end

return MemoryChain

-- END
