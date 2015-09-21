---
layout: post
title:  "Thoughts on torch"
---

I am by no means an expert on torch or lua. But now that I have been playing around with it a bit, I thought it might be worth noting a few initial impressions. In general I am a big fan of torch and to a lesser extend lua. Here I'll explain a bit more why.

### Torch

There are many things torch has going for it:

0. A fantastic tensor library
  0. With good support for in-place modifications of tensors to avoid additional copies. It's possible to get very close to the theoretical lower boundary on how many memory copies are needed for any given algorithm.
0. Great community support (including some Facebook patronage).
0. Ambidextrous use of GPU/CPU.
0. Very powerful and flexible compositional logic for building neural networks.
0. Very little cruft.

In general, I think torch balances the trade-off between how much it is capable without much programming, and how little the "expected way" of doing things doesn't get in the way when you need to implement something custom. This might be visualized like this:

<div class="standard-image">
  <img src="{{"/assets/torch-place.svg" | prepend: site.baseurl }}">
</div>

Basically the only hard thing I need to do when developing with torch is thinking about tensor dimensions. It seems an inordinate amount of my brain cycles are consumed in this way. But I don't fault torch at this, as I think it's an unavoidable aspect of working with multi-dimensional tensors.

### Lua

Lua also has many great things going for it, and by proxy these are also reasons why torch is great:

0. Very fast execution time (very little reason to consider C++ or other compiled languages).
0. Can be easily embedded in other applications.
0. Nice profiler provided by LuaJIT.
0. Given it's interpreted, interactively prototyping code makes it easy to explore how things work.

Unfortunately, there are a number of not so fun aspects of lua:

0. Feels primitive and very bare-bones compared to other scripting languages.
0. Rather unhelpful stack traces.
0. Debugging facilities seem rather lacking.
0. nil. Because variables don't need to be defined, spelling mistakes and other minor errors result in nil, which combined with poor stack traces sometimes makes it hard to locate the problem.
