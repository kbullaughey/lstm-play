local InvertIndex, parent = torch.class('lstm.InvertIndex', 'nn.Module')

-- InvertIndex takes as input a 1D LongTensor of length <= size containing index
-- positions in the range of 1 <= i <= size. It outputs a vector of length
-- size with 1 copied into the indexed cells. If used in batch mode, it can take
-- a table of 1D tensors.

-- size takes the length of the resulting indicator vector
function InvertIndex:__init(size)
  parent.__init(self)
  self.size = size
end

function InvertIndex:updateOutput(indexes)
  if type(indexes) == "table" then
    local batchSize = #indexes
    self.output:resize(batchSize, self.size):zero()
    for i=1,batchSize do
      self.output[i]:indexFill(1, indexes[i], 1)
    end
  else
    self.output:resize(self.size):zero()
    self.output:indexFill(1, indexes, 1)
  end
  return self.output
end
