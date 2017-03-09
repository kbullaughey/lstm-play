-- Create an installer for the namespace
local Installer = function(ns)
  -- Put node on namespace and add debugging connection
  local Install = function(label, inputs, node)
    -- Some nodes don't have inputs, in which case the call signature is (label, node)
    if node == nil then
      if torch.isTypeOf(inputs, "nn.Module") then
        ns[label] = lstm.D(label.."->")(node(lstm.D("->"..label)()))
      else
        error("two-argument form must have a second param derived from nn.Module")
      end
    else
      local annotations = {label=label}
      ns[label] = lstm.D(label.."->")(node(lstm.D("->"..label)(inputs)):annotate(annotations))
    end
  end
  return Install
end

return Installer
