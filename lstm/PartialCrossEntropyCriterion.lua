local PartialCrossEntropyCriterion, Criterion =
  torch.class('lstm.PartialCrossEntropyCriterion', 'nn.Criterion')

function PartialCrossEntropyCriterion:__init(vocabSize)
  Criterion.__init(self)
  self.vocabSize = vocabSize
  self.lsm = nn.LogSoftMax()
  self.nll = nn.ClassNLLCriterion()
  self.inversePartition = torch.Tensor(vocabSize)
  self:resetPartition()
end

-- This parallels the call of setPartition to PartialLinear. It should be called 
-- each time setPartition is called on the associated PartialLinear module.
function PartialCrossEntropyCriterion:setPartition(partition)
  -- We need to translate the indices of target, which are for the original map
  -- size to the indices of the restricted partition.
  -- Since the PartialLinear is only used during training, we should always
  -- have the real answer among the indices.
  self.inversePartition:zero()
  self.inversePartition:indexCopy(1, partition:long(), torch.range(1, partition:size(1)))
end

-- This parallels the call of resetPartition to PartialLinear. It should be called 
-- each time resetPartition is called on the associated PartialLinear module.
function PartialCrossEntropyCriterion:resetPartition()
  self.inversePartition:range(1, self.vocabSize)
end

function PartialCrossEntropyCriterion:updateOutput(input, target)
  input = input:squeeze()
  target = target:squeeze():long()
  self.lsm:updateOutput(input)
  self.nll:updateOutput(self.lsm.output, self.inversePartition:index(1,target))
  self.output = self.nll.output
  return self.output
end

function PartialCrossEntropyCriterion:updateGradInput(input, target)
  local size = input:size()
  input = input:squeeze()
  target = target:squeeze():long()
  self.nll:updateGradInput(self.lsm.output, self.inversePartition:index(1,target))
  self.lsm:updateGradInput(input, self.nll.gradInput)
  self.gradInput:view(self.lsm.gradInput, size)
  return self.gradInput
end

return nn.PartialCrossEntropyCriterion
