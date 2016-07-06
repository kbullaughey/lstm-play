-- Only regularize linear transform weight parameters, not bias parameters. I make
-- this judgement based on whether the parameter tensor is 1D or 2D.
local RegularizationMask = function(net)
  local mask = torch.zeros(net.par:size(1))
  local params = net:parameters()
  local offset = 1
  for i=1,#params do
    local dims = params[i]:dim()
    if dims == 2 then
      local mx_size = params[i]:size(1) * params[i]:size(2)
      mask:narrow(1, offset, mx_size):fill(1)
      offset = offset + mx_size
    else
      local len = params[i]:size(1)
      offset = offset + len
    end
  end
  if offset-1 ~= net.par:size(1) then
    error("unexpected length")
  end
  return mask
end

return RegularizationMask
