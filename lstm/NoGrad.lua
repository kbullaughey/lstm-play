NoGrad = function(node)
  node.data.module.updateGradInput = nn.Module.updateGradInput
  return node
end

return NoGrad
