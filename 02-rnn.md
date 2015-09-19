---
layout: page
title: RNN
longTitle: Recurrent neural network
permalink: /rnn/
---

A major limitation of the feed-forward MLP architecture is that all examples must have the same width. This limitation can be overcome using various recurrent architectures. 

In this example, we'll build a single-layer recursive neural net. We'll reuse the simulated toy data that we trained the MLP on, which happen to all have the same width, but generally recurrent neural nets will be used on sequences of varying lengths. 

### nngraph

We did reasonably well using the stock <code>nn</code> library to build a simple MLP. However, for more complicated architectures, the <code>nngraph</code> package offers an additional layer abstraction that proves quite useful. If you're not familiar with <code>nngraph</code> you may want to see [my brief introduction]({% post_url 2015-09-18-introduction-to-nngraph %}).

{% highlight lua %}
function addUnit(prev_h, x, inputSize, hiddenSize)
  local ns = {}
  -- Concatenate x and prev_h into one input matrix. x is a Bx1 vector and
  -- prev_h is a BxH vector where B is batch size and H is hidden size.
  ns.phx = nn.JoinTable(2,2)({prev_h,x})
  -- Feed these through a combined linear map and squash it.
  ns.h = nn.Tanh()(nn.Linear(inputSize+hiddenSize, hiddenSize)({ns.phx}))
  return ns
end

-- Build the network
function buildNetwork(inputSize, hiddenSize, length)
  -- Keep a namespace.
  local ns = {inputSize=inputSize, hiddenSize=hiddenSize, length=length}

  -- This will be the initial h (probably set to zeros)
  ns.initial_h = nn.Identity()()

  -- This will be expecting a matrix of size BxLxI where B is the batch size,
  -- L is the sequence length, and I is the number of inputs.
  ns.inputs = nn.Identity()()
  ns.splitInputs = nn.SplitTable(2)(ns.inputs)

  -- Iterate over the anticipated sequence length, creating a unit for each
  -- timestep.
  local unit = {h=ns.initial_h}
  for i=1,length do
    local x = nn.Reshape(1)(nn.SelectTable(i)(ns.splitInputs))
    unit = addUnit(unit.h, x, inputSize, hiddenSize)
  end

  -- Output layer
  ns.y = nn.Linear(hiddenSize, 1)(unit.h)

  -- Combine into a single graph
  local mod = nn.gModule({ns.initial_h, ns.inputs},{ns.y})

  -- Set up parameter sharing. The parameter tables will each contain 2L+2
  -- tensors. Each of the first L pairs of tensors will be the linear map
  -- matrix and bias vector for the recurrent units. We'll link each of
  -- these back to the corresponding first one. 
  ns.paramsTable, ns.gradParamsTable = mod:parameters()
  for t=2,length do
    -- Share weights matrix
    ns.paramsTable[2*t-1]:set(ns.paramsTable[1])
    -- Share bias vector
    ns.paramsTable[2*t]:set(ns.paramsTable[2])
    -- Share weights matrix gradient estimate
    ns.gradParamsTable[2*t-1]:set(ns.gradParamsTable[1])
    -- Share bias vector gradient estimate
    ns.gradParamsTable[2*t]:set(ns.gradParamsTable[2])
  end

  -- These vectors will be the flattened vectors.
  ns.par, ns.gradPar = mod:getParameters()
  mod.ns = ns
  return mod
end

{% endhighlight %}

<div class="standard-image">
  <img src="{{"/assets/rnn/model-1_layer-1.png" | prepend: site.baseurl }}">
</div>

<div class='next-page text-center'>
  up next: <a class='page-link' href="{{ "/lstm/" | prepend: site.baseurl }}">LSTM</a>
</div>
