-- Controlling the debugging output globally is achieved through the
-- lstm.debugger interface.

local stringx = require 'pl.stringx'

local debugger = {
  enabled = false,
}

function debugger:enable()
  self.enabled = true
end

function debugger:disable()
  self.enabled = false
end

function debugger:isTensor(thing)
  if string.match(torch.type(thing), "^torch.[A-Za-z]*Tensor") ~= nil then
    return true
  else
    return false
  end
end

function debugger:describeTable(x)
  local lines = {"{"}
  for i,val in ipairs(x) do
    local line
    -- Don't recursivly describe tables
    if type(val) == "table" then
      line = tostring(val)
    else
      line = self:describe(val)
    end
    line = "  [" .. i .. "]: " .. line
    table.insert(lines, line)
    if i == 10 and #x > 10 then
      local remaining = #x - 10
      table.insert(lines, "  ... (skipping " .. remaining .. " entries)")
      break
    end
  end
  table.insert(lines, "}")
  return stringx.join("\n", lines)
end

function debugger:describeNumber(x)
  return "scalar: " .. x
end

function debugger:describeTensor(x)
  local size = stringx.join(",", x:size():totable())
  return torch.type(x) .. ", size: " .. size
end

function debugger:describe(thing)
  local t = torch.type(thing)
  if t == "table" then
    return self:describeTable(thing)
  elseif t == "number" then
    return self:describeNumber(thing)
  elseif self:isTensor(thing) then
    return self:describeTensor(thing)
  elseif t == "nil" then
    return "nil"
  else
    error("Unexpected type: " .. t)
  end
end

return debugger
