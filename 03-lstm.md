---
layout: page
title: LSTM
longTitle: Long-term short-term memory
permalink: /lstm/
---

Now we get to LSTMs, which was my target in teaching myself Torch, Lua, and the <code>nn</code> and <code>nngraph</code> libraries.
My [LSTM implementation](https://github.com/kbullaughey/lstm-play/tree/master/lstm) is based on [code](https://github.com/wojzaremba/lstm) provided in conjunction with [Learning to Execute](http://arxiv.org/abs/1410.4615) paper by Wojciech Zaremba and Ilya Sutskever.
But my main initial inspiration for learning LSTMs came from Andrej Karpathy blog post, [The unreasonable effectiveness of recurrent nerual networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/). His [code base](https://github.com/karpathy/char-rnn) is also a derivative of the Learning to Execute code.

LSTMs supposedly have a major advantage in that they can capture long-term dependencies in the sequence. In the context of NLP this can mean learning structures like matching paraenthesis or brackets hundreds or thousands of characters apart. In addition to the above papers, other applications include, [machine translation](http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf), [captioning images](http://arxiv.org/pdf/1411.4555v2.pdf), handwriting recognition, and much more.

### Adapting our toy

Although standard RNNs can also model sequences or arbitrary length, I have delayed adapting our toy model for that purpose until now. Now we will model sequences of either 2, 3, or 4 inputs, in approximately equal proportions.

Torch is first and foremost a tensor library and efficient computation on modern hardware (GPU or otherwise) requires nice, non-raggad matricies. But this is potentially a problem if we want to operate stochastic gradient descent in mini-batch mode, as each sequence will be a different length. 

The encoding scheme we will adapt is padding each sequence of inputs with zeros, up to a pre-defined maximum length. This allows us to pre-allocate a chain of LSTM units of this length, enabling us to rely on their persisted internal state during forward and backward propagation.

We will also need to pass a vector of the batch-members' sequence lengths. This is impotant for not treating the zero padding as actual inputs, and from injecting the error signal at the right unit in the sequence during back propagation.

We will represent this as an additional set of inputs. Instead of passing <code>mod:forward(input)</code> a single tensor of inputs, we will pass a table containing the inputs and the lengths: <code>mod:forward({inputs, lengths})</code>.

A full training example (including targets) looks like this:
<pre>
th> trainingDataset[1]
{
  1 : 
    {
      1 : DoubleTensor - size: 10x4x1
      2 : DoubleTensor - size: 10
    }
  2 : DoubleTensor - size: 10x1x1
}
</pre>

The inputs look like this (zero padded on the right):
<pre>
th> trainingDataset[1][1][1]:view(10,4)
-1.0481  1.8107  1.0126 -0.0000
-2.6663 -0.3965 -1.4862 -0.4897
-1.0556  1.0742 -0.0000 -0.0000
-0.8559 -1.1087 -1.3144 -0.0000
 0.9741 -1.0469 -2.5899 -0.3083
-0.4533  0.7501 -0.1724 -0.0000
 1.7386  0.6564 -0.3085  1.0727
-2.8677 -0.4960 -1.4441 -0.5026
-0.2834  0.4456  0.3734 -0.0000
 1.4051 -0.3829 -0.0000 -0.0000
[torch.DoubleTensor of size 10x4]
</pre>

And we have these lengths:
<pre>
th> stringx.join(", ", trainingDataset[1][1][2]:totable())
2, 3, 4, 3, 4, 4, 3, 2
</pre>

### LSTM implementation

I have packaged up my LSTM implementation into a luarock, lstm, available on [github](https://github.com/kbullaughey/lstm-play/tree/master/lstm). Like most other folks in the field, I have adoped [Alex Graves's LSTM formulation](http://arxiv.org/pdf/1308.0850v5.pdf) and reference his equation numbers in the source below. Here's a screenshot from his paper:

<div class="full-width-image">
  <img src="{{"/assets/lstm/graves_equations.png" | prepend: site.baseurl }}">
</div>

I adopt an optimization used in the Learning to Execute [implementation](https://github.com/wojzaremba/lstm), which is to change the order of operations so as to perform fewer matrix multiplications (albeit on larger matrices). The number of parameters and the number of multiplications doesn't change, but by packaging them into fewer matrix multiplications we can get somewhat better performance on modern hardware. For example, equations (7) and (8), involve mapping x<sub>t</sub>, h<sub>t-1</sub>, and c<sub>t-1</sub> through linear maps and then adding the resulting vectors. The same can be accomplished by first concatenating the three vectors and then mapping them through an appropriately larger parameter matrix.

Creating a memory cell is substantially more involved than an RNN unit:

{% highlight lua %}
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

  -- In two cases, we use sums like W_c * prev_c, we use one linear map for these
  -- cases and then split. The inputs to this map will have dimension BxQ and the
  -- outputs will have dimension Bx2Q. We reshape this to Bx2xQ and split into two
  -- BxQ streams.
  local prev_c2h = nn.Linear(hidden_size, hidden_size*2)(prev_c)
  local prev_c2h_reshaped = nn.Reshape(2, hidden_size, true)(prev_c2h)
  local prev_c2h_split_by_gate = nn.SplitTable(2)(prev_c2h_reshaped)

  -- Separate out the split tables for the linear maps involving the memory.
  local prev_c2h_i_gate   = nn.SelectTable(1)(prev_c2h_split_by_gate)
  local prev_c2h_f_gate   = nn.SelectTable(2)(prev_c2h_split_by_gate)

  -- Compute the gate values
  local i_gate = nn.Sigmoid()(nn.CAddTable()({xh2h_i_gate, prev_c2h_i_gate})) -- Eq. (7)
  local f_gate = nn.Sigmoid()(nn.CAddTable()({xh2h_f_gate, prev_c2h_f_gate})) -- Eq. (8)

  -- Update the memory, Eq. (9)
  local c = nn.CAddTable()({
    nn.CMulTable()({f_gate, prev_c}),
    nn.CMulTable()({i_gate, nn.Tanh()(xh2h_learning)})
  })

  local c2h_o_gate = nn.Linear(hidden_size, hidden_size)(c)
  local o_gate = nn.Sigmoid()(nn.CAddTable()({xh2h_o_gate, c2h_o_gate})) -- Eq. (10)
  
  -- Squash the memory and then mask it with the output gate. Eq. (11)
  local h = nn.CMulTable()({o_gate, nn.Tanh()(c)})

  -- Make the module encompasing the whole LSTM
  return nn.gModule({x, prev_h, prev_c}, {h,c})
end
{% endhighlight %}

The crux of the code to string memory cells gether is rather straightforward, it is parameterized by <code>inputSize</code> and <code>hiddenSizes</code>, and handles creating an LSTM with one or more layers:

{% highlight lua %}
-- Here I create tables that will store the shared parameters, one for
-- forward parameters and one for the accumulating gradient estimates.
self.lstmParams = {}
self.lstmGradParams = {}

self.lstms = {}
local numLayers = #hiddenSizes
local prevLayerSize = inputSize
for l=1,numLayers do
  local thisLayerSize = hiddenSizes[l]
  self.lstms[l] = {}

  -- Make enough lstm cells for the longest sequence
  for t=1,maxLength do
    self.lstms[l][t] = lstm.MemoryCell(prevLayerSize, thisLayerSize)
  end

  -- Capture the parameters of the first cell in this layer, as these will
  -- be shared across the layer.
  self.lstmParams[l], self.lstmGradParams[l] = self.lstms[l][1]:parameters()

  -- The output of one layer is the input to the next.
  prevLayerSize = thisLayerSize
end
{% endhighlight %}

And we'll also need to share parameters among memory cells:
{% highlight lua %}
-- Share parameters among all memory cells of each layer. Parameters are not
-- shared between layers.
function MemoryChain:share()
  -- The first cell in each layer is the reference parameters. We'll share all
  -- subsequent cells in the layer back with this first one.
  for l=1,self.numLayers do

    -- Get the shared parameters for this layer.
    local sharedParams = self.lstmParams[l]
    local sharedGradParams = self.lstmGradParams[l]

    for t=2,self.maxLength do
      -- Get the parameters for the memory cell in layer l at timestep t. This
      -- will be a table containing the parameters for each unit in the LSTM
      -- MemoryCell module.
      local cellParams, cellGradParams = self.lstms[l][t]:parameters()

      -- Iterate over each component's parameters, setting them to use the
      -- memory of the reference memory cell, which we captured during setup.
      for i=1, #cellParams do
        cellParams[i]:set(sharedParams[i])
        cellGradParams[i]:set(sharedGradParams[i])
      end
    end
  end
end
{% endhighlight %}

### Training and performance

LSTMs have many more parameters than our simple RNN, mainly do to all the additional parameter matricies involved in the input, forget, and output gates, and also because they have both a hidden state and a memory cell. So where our RNN with <code>h=26</code> had 755 parameters, our LSTM with <code>h=16</code> has 2049.

The performance looks very good:

<div class="standard-image">
  <img src="{{"/assets/lstm/model-1_layer-1.png" | prepend: site.baseurl }}">
</div>


