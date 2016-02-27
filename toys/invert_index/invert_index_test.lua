require 'lstm'

node = lstm.InvertIndex(4)

-- Test forward

indexes = torch.LongTensor{1,4,2}
res = node:forward(indexes)

if res:eq(torch.Tensor{1,1,0,1}):sum() ~= 4 then
  error("inverting didn't get expected response")
end

-- Test backward

node:backward(indexes, torch.Tensor{1,2,3,4})

if node.gradInput:dim() ~= 0 then
  error("expecting dimensionless gradient")
end

-- It can work in batch mode.
indexes = {torch.LongTensor{1,4,2}, torch.LongTensor{2,3}}
res = node:forward(indexes)

if res:eq(torch.Tensor{{1,1,0,1},{0,1,1,0}}):sum() ~= 8 then
  error("inverting didn't get expected response")
end
