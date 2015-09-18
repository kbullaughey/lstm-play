---
layout: page
title: MLP
longTitle: Multi-layer perceptron
permalink: /mlp/
---

### Introduction to nerual nets with torch

Torch includes a very elegant collection of abstractions for building neural networks of various topologies ([torch/nn](https://github.com/torch/nn)). Because the components have a relatively uniform interface and fit together in standard ways, it straightforward to make all sorts of network topologies that
<em>Just Work&trade;</em>.

Each component compartentalizes the details of how outputs are computed from inputs and how to back-propagate error signals. Thus when using pre-built components one merely needs to connect them together and all the details of forward and backward propagation are handled automatically and (usually) without error. In my experience I rarely have trouble with my gradients not being correct given the topology, my errors are usually in specifying the incorrect topology (or other issues like whether parameter sharing is set up properly, etc.).

> The <span class='fw'>nn</span> modules are more than neural net components, they are powerful, general purpose tools for computing with matricies and tables of matricies.

Here's a sampling computations for which there are <span class='fw'>nn</span> modules:

0. linear transformation
0. component-wise multiplication, or division
0. addition of or multiplication a scalar
0. propagate only a subset of columns or rows, reshape a tensor, or replicate inputs
0. max, min, mean, sum, exp, abs, power, sqrt, square, normalize, etc.
0. dropout
0. join, split, subset or flatten tables
0. mixture of experts
0. dot product, cosine distance
0. element-wise addition, subtraction, multiplication, division of tensors in a table
0. convolution of all sorts

I will use the module for doing a linear map as an example. I will use a context (computer graphics) to illustrate that these modules can be used for general purpose computation, not just nerual nets.

Here's a matrix that represents clockwise rotation by 90&deg; and a point at coordinates (2,0):

<pre>
th> rotate = torch.Tensor{0,1,0,-1,0,0,0,0,1}
 0  1  0
-1  0  0
 0  0  1
[torch.DoubleTensor of size 3x3]
th> x = torch.Tensor{2,0,1}
 2
 0
 1
[torch.DoubleTensor of size 3]
</pre>

We can then declare a linear map nn module, copy in the parameters for our linear map (setting the bais to zero):

{% highlight lua %}
m = nn.Linear(3,3)
m:parameters()[1]:copy(rotate)
m:parameters()[2]:zero()
{% endhighlight %}

Now, when we do forward propagation, we simply performing matrix multiplication, in this case, causing our vector to rotate 90&deg; clockwise:

<pre>
th> m:forward(x)
 0
-2
 1
[torch.DoubleTensor of size 3]
</pre>

### A classic feed-forward, MLP

Setting up a basic neural net is very simple:

{% highlight lua %}
num_inputs = 3
-- I pick this odd hidden layer size to result in the same number of parameters
-- as the two-layer example later.
h_size = 152
mlp = nn.Sequential()
mlp:add(nn.Linear(num_inputs, h_size))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(h_size, 1))
{% endhighlight %}

The <span class='fw'>nn</span> package contains a number of objective functions, including mean squared error and cross entropy. Since the target of our <a href="{{ "/toy/" | prepend: site.baseurl }}">toy model</a> is real-valued, we'll use least-square loss:

{% highlight lua %}
criterion = nn.MSECriterion()
{% endhighlight %}

Now we can set up a trainer to use stochastic gradient descent:

{% highlight lua %}
-- Set up a SGD trainer.
trainer = nn.StochasticGradient(mlp, criterion)
trainer.maxIteration = 50
{% endhighlight %}

We can then train the model, first setting initial parameters:

{% highlight lua %}
-- Get all the parameters packaged into a vector so we can reset them. It is important
-- to save a reference to this memory, because each call of getParameters() may
-- put the memory into new storage location.
mlp_params = mlp:getParameters()
mlp_params:uniform(-0.1, 0.1)
-- Train the model, after randomly initializing the parameters and clearing out
-- any existing gradient.
trainer:train(dataset)
{% endhighlight %}

The above model has 761 parameters. I [run](https://github.com/kbullaughey/lstm-play/blob/master/toys/mlp/model-1_layer.sh) it with a batch size of 20, 70,000 training examples, a learn rate of 0.05, and 50 epoch ([full code](https://github.com/kbullaughey/lstm-play/blob/master/toys/mlp/model-1_layer.lua)). Despite having lots of parameters and plenty of time to converge, it doesn't do real well:

<div class="standard-image">
  <img src="{{"/assets/mlp/model-1_layer-1.png" | prepend: site.baseurl }}">
</div>

[Yoshua Bengio suggests](http://arxiv.org/pdf/1206.5533v2.pdf) that we should pick the highest learn rate that doesn't result in divergence, and that generally this is within a factor of two of the optimal learn rate. For the above, I selected a learn rate of 0.05. If we go just a little bit higher, to 0.06, all hell breaks lose:

<div class="standard-image">
  <img src="{{"/assets/mlp/model-1_layer-fail-1.png" | prepend: site.baseurl }}">
</div>

It's a very straightforward extension to add a second layer:

{% highlight lua %}
num_inputs = 3
h1_size = 30
h2_size = 20
mlp = nn.Sequential()
mlp:add(nn.Linear(num_inputs, h1_size))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(params.h1, h2_size))
mlp:add(nn.Sigmoid())
mlp:add(nn.Linear(h2_size, 1))
{% endhighlight %}

This model also has 761 parameters. I train it with the same training data, batch size (20), learn rate (0.05) and number of epochs (50) as before. Interestingly, this performs much better, with just a few places where it gets confused:

<div class="standard-image">
  <img src="{{"/assets/mlp/model-2_layer-1.png" | prepend: site.baseurl }}">
</div>

While a classic feed-forward, multi-layer perceptron performs well on this toy problem, it's not very flexible because it can't handle varying-length sequences.

<div class='next-page text-center'>
  up next: <a class='page-link' href="{{ "/rnn/" | prepend: site.baseurl }}">RNN</a>
</div>
