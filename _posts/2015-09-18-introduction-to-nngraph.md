---
layout: post
title:  "Introduction to nngraph"
---

The <code>nngraph</code> overloads <code>nn.Module</code> (which actually are tables) so that they can be called as functions (using a language feature called metamethods). These constructor-like functions are used to link Modules together in a function-compositional way than is much more natural than ordinarily possible with standard <code>nn.Modules</code>.

I will illustrate with a few examples that we will later use to build an RNN. The simplest <code>nn.Module</code> is the <code>nn.Identity</code> module that simply feeds its input forward, unchanged and it has no parameters:

<pre>
th> x = nn.Identity()
th> x:forward(2)
2
th> x:backward(2,3)
3
</pre>

Suppose for some crazy reason we want to chain several of these together. This can be accomplished using standard <code>nn</code> Modules as follows:

{% highlight lua %}
seq = nn.Sequential()
seq:add(nn.Identity())
seq:add(nn.Identity())
seq:add(nn.Identity())
{% endhighlight %}

This behaves as before: 
<pre>
th> seq:forward(2)
2
th> seq:backward(2,3)
3
</pre>

However, with <code>nngraph</code> we can accomplish this in a compositional way:

{% highlight lua %}
first = nn.Identity()()
twoMore = nn.Identity()(nn.Identity()(first))
seq = nn.gModule({first},{twoMore})
{% endhighlight %}

What's going on here is we're saving a reference to the input, <code>first</code>, then compositionally chaining together two more in the sequence, and finally using the <code>nn.gModule</code> method provided by <code>nngraph</code> to bundle the inputs and outputs into a single Module that works as a single unit.

And we can use it as before:

<pre>
th> seq:forward(2)
2
th> seq:backward(2,3)
3
</pre>

While that didn't seem any easier, the compositional nature starts to pay off once the architecture gets more involved.

{% highlight lua %}
ni = 2; nh = 4; no = 1; len = 5
h0 = nn.Identity()()
x = nn.Identity()()
xs = nn.SplitTable(2)(x)
h = h0
for i=1,len do
  h = nn.Tanh()(nn.Linear(ni+nh,nh)(nn.JoinTable(1)({h,nn.SelectTable(1)(xs)})))
end
y = nn.Linear(nh,no)(h)
rnn = nn.gModule({h0,x},{y})
{% endhighlight %}

This builds a single-layer RNN with 5 timesteps input dimension <code>ni</code>, hidden state size <code>nh</code>, and output dimensionality <code>no</code>. For more details on RNNs and a detailed walkthough on what this sort of code does, have a look at the [RNN]({{ "/rnn/" | prepend: site.baseurl }}) part of the tutorial.

The code to build this without <code>nngraph</code> would have been much more complicated.

