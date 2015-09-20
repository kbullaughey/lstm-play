---
layout: page
title: LSTM
longTitle: Long-term short-term memory
permalink: /lstm/
---

Now we get to LSTMs, which was my target in teaching myself Torch, Lua, and the <code>nn</code> and <code>nngraph</code> libraries.
My LSTM implementation is based on [code](https://github.com/wojzaremba/lstm) provided in conjunction with [Learning to Execute](http://arxiv.org/abs/1410.4615) paper by Wojciech Zaremba and Ilya Sutskever.
But my main initial inspiration for learning LSTMs came from Andrej Karpathy blog post, [The unreasonable effectiveness of recurrent nerual networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). His [code base](https://github.com/karpathy/char-rnn) is also a derivative of the Learning to Execute code.

LSTMs supposedly have a major advantage in that they can capture long-term dependencies in the sequence. In the context of NLP this can mean learning structures like matching parenthesis or brackets hundreds or thousands of characters apart. In addition to the above papers, other applications include, [machine translation](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), [captioning images](http://arxiv.org/pdf/1411.4555v2.pdf), handwriting recognition, and much more.

### LSTM implementation

Like most other folks in the field, I have adopted [Alex Graves's LSTM formulation](http://arxiv.org/pdf/1308.0850v5.pdf) and reference his equation numbers in the source below. Here's a screenshot from his paper:

<div class="full-width-image">
  <img src="{{"/assets/lstm/graves_equations.png" | prepend: site.baseurl }}">
</div>

Here I will highlight some of my implementation. Please refer to the [full script](https://github.com/kbullaughey/lstm-play/blob/master/toys/lstm/model-1_layer.lua) to see how data is loaded and training is done.

Creating a memory cell is substantially more involved than an RNN unit:

{% highlight lua %}
function addUnit(prev_h, prev_c, x, inputSize, hiddenSize)
  local ns = {}
  -- Input gate. Equation (7)
  ns.i_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h),
    Linear(hiddenSize,hiddenSize)(prev_c)
  }))
  -- Forget gate. Equation (8)
  ns.f_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h),
    Linear(hiddenSize,hiddenSize)(prev_c)
  }))
  -- New contribution to c. Right term in equation (9)
  ns.learning = nn.Tanh()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h)
  }))
  -- Memory cell. Equation (9)
  ns.c = nn.CAddTable()({
    nn.CMulTable()({ns.f_gate, prev_c}),
    nn.CMulTable()({ns.i_gate, ns.learning})
  })
  -- Output gate. Equation (10)
  ns.o_gate = nn.Sigmoid()(nn.CAddTable()({
    Linear(inputSize,hiddenSize)(x),
    Linear(hiddenSize,hiddenSize)(prev_h),
    Linear(hiddenSize,hiddenSize)(ns.c)
  }))
  -- Updated hidden state. Equation (11)
  ns.h = nn.CMulTable()({ns.o_gate, ns.c})
  return ns
end
{% endhighlight %}

The crux of the code to string memory cells together is rather straightforward, it is parameterized by <code>inputSize</code> and <code>hiddenSizes</code>, and handles creating an LSTM with one or more layers:

{% highlight lua %}
-- This will be the initial h and c (probably set to zeros)
initial_h = nn.Identity()()
initial_c = nn.Identity()()

-- This will be expecting a matrix of size BxLxI where B is the batch size,
-- L is the sequence length, and I is the number of inputs.
inputs = nn.Identity()()
splitInputs = nn.SplitTable(2)(inputs)

-- Iterate over the anticipated sequence length, creating a unit for each
-- timestep.
local unit = {h=initial_h, c=initial_c}
for i=1,length do
  local x = nn.SelectTable(i)(splitInputs)
  unit = addUnit(unit.h, unit.c, x, inputSize, hiddenSize)
end

-- Output layer
y = Linear(hiddenSize, 1)(unit.h)

-- Combine into a single graph
mod = nn.gModule({initial_h, initial_c, inputs},{y})
{% endhighlight %}

And we'll also need to share parameters among memory cells. You might have noticed I've been using a global function, <code>Linear</code>, but the name is not prefixed with <code>nn</code>, but it seems to work the same as <code>nn.Linear</code>. This is for the purpose of setting up parameter sharing. My function masquerading as <code>nn.Linear</code> keeps a table of all the linear modules while constructing the net, so that we have references to them later to set up parameter sharing:

{% highlight lua %}
LinearMaps = {}
Linear = function(a, b)
  local mod = nn.Linear(a,b)
  table.insert(LinearMaps, mod)
  return mod
end
{% endhighlight %}

We can then use this list to tie the 11 Linear maps of each Memory cell back to the first one:

{% highlight lua %}
-- Set up parameter sharing. 
for t=2,numInput do
  for i=1,11 do
    local src = LinearMaps[i]
    local map = LinearMaps[(t-1)*11+i]
    local srcPar, srcGradPar = src:parameters()
    local mapPar, mapGradPar = map:parameters()
    if #srcPar ~= #mapPar or #srcGradPar ~= #mapGradPar then
      error("parameters structured funny, won't share")
    end
    mapPar[1]:set(srcPar[1])
    mapPar[2]:set(srcPar[2])
    mapGradPar[1]:set(srcGradPar[1])
    mapGradPar[2]:set(srcGradPar[2])
  end
end
{% endhighlight %}

### Training and performance

LSTMs have many more parameters than our simple RNN, mainly do to all the additional parameter matrices involved in the input, forget, and output gates, and also because they have both a hidden state and a memory cell. So where our RNN with <code>h=26</code> had 755 parameters, our LSTM with <code>h=16</code> has 2049.

The performance looks very good:

<div class="standard-image">
  <img src="{{"/assets/lstm/model-1_layer-1.png" | prepend: site.baseurl }}">
</div>


