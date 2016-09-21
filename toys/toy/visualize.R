# Here we will simulate an arbitrary relation from Real -> Real. We will try and
# learn the inverse. The function will wavy, and thus the inverse will be one to
# many. Thus, simply learning the mapping based on one input and one output is
# impossible. Instead, we will incorporate a few adjacent inputs, which
# collectively should have enough information to identify the correct output.

# Make this reproducible.
set.seed(1)

# An arbitrary, wavy function.
f <- function(x) 2 + 0.5*sin(x/2 - 1) + sin(x) + sin(x*2 + 2) + sin(x/4 + 1)

# The domain of the forward mapping is (-2,10)
x_min <- -2
x_max <- 10

# For plotting purposes, we'll use a grid of points.
x_grid <- seq(x_min, x_max, length.out=100)
y_grid <- f(x_grid)
y_max <- max(y_grid)
y_min <- min(y_grid)

# To make the problem a bit harder, we'll add some noise.
noise <- 0.4

# Plot several variations of a figure. This function accepts a list of
# closures and a list of masks indicating which closures to print together.
# Each mask is a list of indices into the closures list.
plot_staged_figures <- function(closures, masks) {
  lapply(1:length(masks), function(i) {
    lapply(closures[masks[[i]]], function(func) {
      do.call(func, list())
    })
  })
  return()
}

#---------
# Figure 1
#---------

# Plot the inverse mapping
pdf(file="figure_1.pdf", height=4, width=4)
par(mar=c(4,4,1,1), mgp=c(2,0.8,0), cex.lab=1, cex.axis=0.8)
plot(y_grid, x_grid, type="l", xlab="input", ylab="output")
abline(v=2.6)
dev.off()

#---------
# Figure 2
#---------

# Illustrate how sequential information can help us solve the multi-map problem
example_x <- 6.61029
example_3x <- example_x - 0:2
example_y <- as.numeric(f(example_3x))
example_y_observed <- example_y + rnorm(3, mean=0, sd=noise)

pdf(file="figure_2.pdf", height=4, width=4)
par(mar=c(4,4,1,3), mgp=c(2,0.8,0), cex.lab=1, cex.axis=0.8, xpd=NA)
plot(y_grid, x_grid, type="l", xlab="input", ylab="output")
segments(y_min, example_x, y_max, example_x, col="firebrick", lty=3)
points(example_y_observed, example_3x)
text(y_max+0.5, example_x, labels="z", cex=0.7, col="firebrick")
points(example_y_observed, rep(x_min,3), pch=17, col="steelblue")
text(example_y_observed, rep(x_min+0.5,3), labels=as.character(1:3), cex=0.6, col="steelblue")
text(example_y_observed-0.5, example_3x, col="steelblue")
par(xpd=FALSE)
abline(v=-3, col="gray60")
dev.off()

#---------
# Figure 3
#---------

# Correlate a bunch of examples inputs and the outputs we try and predict to
# give a better taste of the prediction problem.

# Evenly space the examples.
n_ex <- 10
examples_x <- seq(x_min+2, x_max, length.out=n_ex)
examples_3x <- t(sapply(examples_x, function(x) x - 0:2))
examples_y <- t(apply(examples_3x, 1, f))
examples_y_observed <- examples_y + rnorm(3, mean=0, sd=noise)

# This figure has two panels. On the left I present the inputs, organized as
# rows with the three inputs per row. I permute the input rows so the problem
# is presented more realistically because when actually doing prediction we
# won't know anything about the vertical coordinates, which correspond to the
# outputs we're trying to predict. On the right I correspond these to the 
# outputs. 
perm <- sample(1:n_ex, n_ex, replace=FALSE)
near_right <- max(examples_y_observed) + 0.2
far_right <- near_right + 1.1
left <- min(examples_y_observed)
transparent_firebrick <-
  do.call(rgb, as.list(c(as.numeric(col2rgb("firebrick")/255), 0.3)))

pdf(file="figure_3.pdf", height=4, width=8)
plot_staged_figures(list(
  function() {
    par(mgp=c(2,0.8,0), cex.lab=1, cex.axis=0.8, xpd=NA, mfcol=c(1,2))
    par(mar=c(4,2,1,1))
    plot(c(left, near_right), range(x_grid), type="n", axes=FALSE,
      xlab="inputs", ylab="")
    axis(1)
  },function() {
    segments(rep(left,n_ex), examples_x[perm], rep(near_right,n_ex), examples_x[perm],
      col=transparent_firebrick, lty=2)
  },function() {
    trash <- lapply(1:3, function(i) {
      points(examples_y_observed[,i], examples_x[perm], pch=as.character(i),
        col="steelblue", cex=0.7)
    })
    par(mar=c(4,1,1,4))
  },function() {
    segments(near_right, examples_x[perm], far_right, examples_x, col="firebrick")
  },function() {
    points(rep(near_right,n_ex), examples_x[perm], pch=20, cex=0.5, col="firebrick")
    points(rep(far_right,n_ex), examples_x, pch=20, cex=0.5, col="firebrick")
    plot(range(y_grid), range(x_grid), type="n", xlab="", ylab="", axes=FALSE)
    lines(y_grid, x_grid)
    axis(4)
    mtext("outputs", side=4, line=2)
  },function() {
    segments(rep(y_min,n_ex), examples_x, rep(y_max,n_ex), examples_x,
      col=transparent_firebrick, lty=2)
  },function() {
    mtext("mapping to learn", side=2, line=0.2, col="firebrick", cex=0.9)
  }), masks=list(
    c(1,2,3,5,7),
    c(1,2,3,4,5,6)
  )
)
dev.off()

#-------------------
# Figures 4a, 4b, 4c
#-------------------

n <- 100
x <- seq(0, x_max, length.out=n)
y <- f(cbind(x, x-1, x-2))
idx <- 1:n

# 4a
single_euclid_dist <- apply(expand.grid(y[,1], y[,1]), 1, function(yy) abs(yy[1] - yy[2]))
single_sim <- matrix(max(single_euclid_dist) - single_euclid_dist, ncol=n)
pdf(file="figure_4a.pdf", height=4, width=4)
par(mar=c(1,1,1,1))
image(single_sim[n:1,], axes=FALSE, col=gray.colors(12))
box()
dev.off()

# 4b
euclid_dist <- apply(expand.grid(idx, idx), 1, function(p) sqrt(sum((y[p[1],]-y[p[2],])^2)))
similarity <- matrix(max(euclid_dist) - euclid_dist, ncol=n)
pdf(file="figure_4b.pdf", height=4, width=4)
par(mar=c(1,1,1,1))
image(similarity[n:1,], axes=FALSE, col=gray.colors(12))
box()
dev.off()

# 4c
# Now add noise and reproduce Figure 4
yn <- y + rnorm(n*3, mean=0, sd=noise)
euclid_dist_n <- apply(expand.grid(idx, idx), 1, function(p) sqrt(sum((y[p[1],]-yn[p[2],])^2)))
similarity_n <- matrix(max(euclid_dist_n) - euclid_dist_n, ncol=n)
pdf(file="figure_4c.pdf", height=4, width=4)
par(mar=c(1,1,1,1))
image(similarity_n[n:1,], axes=FALSE, col=gray.colors(12))
box()
dev.off()

# END
