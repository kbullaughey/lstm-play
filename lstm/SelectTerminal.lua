-- This module enables one to use batch mode with sequences of different lengths.
-- It takes two inputs, the first is a tensor of size BxLxH and the second is a
-- vector of length B, where B is the batch size, H is the length of the LSTM's
-- hidden state and L is the longest possible sequence. The vector contains the
-- sequence length of each batch item. It's used to select the appropriate column
-- from the first input. The output is a BxH matrix with each row the column selected
-- from the first input.
local SelectTerminal, parent = torch.class('lstm.SelectTerminal', 'nn.Module')

function SelectTerminal:__init()
  parent.__init(self)
  self.scratchA = torch.Tensor()
  self.scratchB = torch.Tensor()
  self.gradInput = {self.scratchA, self.scratchB}
end

function SelectTerminal:updateOutput(tuple)
  local input, lengths = self:splitTuple(tuple)
  local B = input:size(1)
  local L = input:size(2)
  local H = input:size(3)
  if lengths:size(1) ~= B then
    error("Vector of lengths should match batch size (first dimension of input)") 
  end
  self.output:resize(B,H)
  for b=1,B do
    local lastHidden = lengths[b]
    self.output[b]:copy(input[b][lastHidden])
  end
  return self.output
end

function SelectTerminal:updateGradInput(tuple, gradOutput)
  local input, lengths = self:splitTuple(tuple)
  self.scratchA:resizeAs(input):zero()
  self.scratchB:resizeAs(lengths):zero()
  local B = input:size(1)
  for b=1,B do
    local lastHidden = lengths[b]
    self.gradInput[1][b][lastHidden]:copy(gradOutput[b])
  end
  return self.gradInput
end 

function SelectTerminal:splitTuple(tuple)
  local input, lengths = table.unpack(tuple)
  if input:dim() ~= 3 then
    error("Expecting a 3D input tensor")
  end
  if lengths:dim() ~= 1 then
    error("Expecting vector of lengths to be 1D tensor")
  end
  return input, lengths
end

-- END
