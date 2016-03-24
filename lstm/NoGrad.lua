-- NoGrad prevents the gradient from being passed back for parts of the graph near th
-- beginning that have no parameters and are only massaging formats.
-- This is designed to wrap either nngraph.Node or nn.Module-derived instances.
local NoGrad = function(node)
  if torch.type(node) == "nngraph.Node" then
    node.data.module.updateGradInput = nn.Module.updateGradInput
  else
    node.updateGradInput = nn.Module.updateGradInput
  end
  return node
end

return NoGrad
