---
layout: page
title: Toy
longTitle: Toy problem
permalink: /toy/
---

Since we want to understand LSTMs, which are used for modeling sequential data, we
would like to create a toy problem for which sequential information is important to
successfully modeling it.

Here I use a weighted average of five arbitrary functions to come up with a mapping (written in R):

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

In order to add a sequential aspect to this modeling problem we can consider three inputs instead of one. One (arbitrary) way to do this is to consider three evenly spaced points in the original mapping (z, z-1, z-2). Each of these can be transformed by our function with some noise added: f(z) + N(0, 0.16). We then can consider these three values as the inputs, and our goal is to predict what is the original z:

<div class="standard-image">
  <img src="{{"/assets/figure_2.png" | prepend: site.baseurl }}">
</div>

In the above plot, the inputs are marked by the three blue triangles at the bottom. The output we're trying to predict is the red dotted line at z. The inputs are derived by adding two points evenly spaced below z, computing their corresponding f(z) values and adding some noise.

By having access to a sequence of three inputs we now can get a better idea of what part of the curve these must have been derived from. In this way, the sequential information provides context and should enable us to solve the inverse mapping problem. With a single input, we would only know the location of the gray line, which crosses the curve at six places. But with the three ordered triangles, there is much less ambituity that these points must have been derived from z (red dotted line), even despite the additional noise.



