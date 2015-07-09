require 'nngraph'

-- Make an LSTM graph node.
-- 
-- For a batch size, B, input_size, P, and hidden_size, Q, the sizes should
-- be as follows:
--
--    x: BxP
--    prev_h: BxQ
--    prev_c: BxQ
--
-- Returns an nn Module output from nngraph gModule()
local function MemoryCell(input_size, hidden_size)

  -- Input placeholders
  local x = nn.Identity()()
  local prev_h = nn.Identity()()
  local prev_c = nn.Identity()()

  -- There are four sets of weights for each of x, and prev_h. The inputs to
  -- these two linear maps are sizes BxP and BxQ respectively. The outputs are
  -- both Bx4Q.
  local x2h = nn.Linear(input_size, hidden_size*4)(x)
  local h2h = nn.Linear(hidden_size, hidden_size*4)(prev_h)

  -- We're always ading W_x * x + W_h * h, regardless of gate, so we can add
  -- the combined tables.
  local xh2h = nn.CAddTable()({x2h, h2h})

  -- Data flowing through xh2h is size Bx4Q. We reshape this to Bx4xQ so that
  -- that we can separate the data into four separate BxQ streams. Thus when
  -- we split, we split on the second dimension to split into four separate
  -- streams.
  local xh2h_reshaped = nn.Reshape(4, hidden_size, true)(xh2h)
  local xh2h_split_by_gate = nn.SplitTable(2)(xh2h_reshaped)

  -- Separate out the split tables.
  local xh2h_i_gate   = nn.SelectTable(1)(xh2h_split_by_gate)
  local xh2h_f_gate   = nn.SelectTable(2)(xh2h_split_by_gate)
  local xh2h_learning = nn.SelectTable(3)(xh2h_split_by_gate)
  local xh2h_o_gate   = nn.SelectTable(4)(xh2h_split_by_gate)

  -- In three cases, we use sums like W_c * prev_c, we use one linear map for all
  -- cases and then split. The inputs to this map will have dimension BxQ and the
  -- outputs will have dimension Bx3Q. We reshape this to Bx3xQ and split into three
  -- BxQ streams.
  local c2h = nn.Linear(hidden_size, hidden_size*3)(prev_c)
  local c2h_reshaped = nn.Reshape(3, hidden_size, true)(c2h)
  local c2h_split_by_gate = nn.SplitTable(2)(c2h_reshaped)

  -- Separate out the split tables for the linear maps involving the memory.
  local c2h_i_gate   = nn.SelectTable(1)(c2h_split_by_gate)
  local c2h_f_gate   = nn.SelectTable(2)(c2h_split_by_gate)
  local c2h_o_gate   = nn.SelectTable(3)(c2h_split_by_gate)

  -- Compute the gate values
  local i_gate = nn.Sigmoid()(nn.CAddTable()({xh2h_i_gate, c2h_i_gate})) -- Eq. (7)
  local f_gate = nn.Sigmoid()(nn.CAddTable()({xh2h_f_gate, c2h_f_gate})) -- Eq. (8)
  local o_gate = nn.Sigmoid()(nn.CAddTable()({xh2h_o_gate, c2h_o_gate})) -- Eq. (10)

  -- Update the memory, Eq. (9)
  local c = nn.CAddTable()({
    nn.CMulTable()({f_gate, prev_c}),
    nn.CMulTable()({i_gate, nn.Tanh()(xh2h_learning)})
  })
  
  -- Squash the memory and then mask it with the output gate. Eq. (11)
  local h = nn.CMulTable()({o_gate, nn.Tanh()(c)})

  -- Make the module encompasing the whole LSTM
  return nn.gModule({x, prev_h, prev_c}, {h,c})
end

return MemoryCell

-- END
