---
layout: page
title: MLP
longTitle: Multi-layer perceptron
permalink: /mlp/
---

We'll first first attack a toy modeling problem using a multi-layer perceptron, pretty much the simplest type of neural net. Then we'll progress to recursive neuran lets (RNN) and finally the long-term short-term memory (LSTM). But to begin, we'll need a toy problem to model.

## Toy problem

Since we want to understand LSTMs, which are used for modeling sequential data, we
would like to create a toy problem for which sequential information is important to
successfully modeling it.

Here I use a weighted average of five arbitrary functions to come up with a mapping:

{% highlight R %}
# These are arbitrary functions that will be combined to form the forward
# mapping.
f1 <- function(x) sin(x*3)
f2 <- function(x) (x-5)^2
f3 <- function(x) log(x+0.1)
f4 <- function(x) 1+3*x
f5 <- function(x) (x/2-4)^2

# The above functions are averaged using these weights.
weights <- c(1,0.2,-1,-0.1,-0.2)

# Compute the weighted average of the five functions.
f <- function(x, w=weights) {
  z = cbind(f1(x), f2(x), f3(x), f4(x), f5(x))
  (z %*% w) * (1 + 0.2*sin(1.5*x-1))
}
{% endhighlight %}

This will result in a somewhat wavy function, and by considering the inverse mapping, we get a modeling problem whereby a single input (horizontal axis) doesn't uniquely define the output (vertical axis), as illustrated by the vertical line which crosses the curve in six places:

<div class="standard-image">
  <img src="{{"/assets/mlp-mapping.png" | prepend: site.baseurl }}">
</div>

