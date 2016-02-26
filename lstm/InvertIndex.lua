local InvertIndex, parent = torch.class('lstm.InvertIndex', 'nn.Module')

-- InvertIndex takes as input a 1D LongTensor of length <= size containing index
-- positions in the range of 1 <= i <= size. It outputs a vector of length
-- size with 1 copied into the indexed cells.

-- size takes the length of the resulting indicator vector
function InvertIndex:__init(size)
  parent.__init(self)
  self.size = size
end

function InvertIndex:updateOutput(indexes)
  local t = input[1]
  local index = input[2]
  self.output:resize(self.size):zero()
  self.output:indexFill(1, indexes, 1)
  return self.output
end
