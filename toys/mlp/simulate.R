# Here we will simulate an arbitrary relation from Real -> Real. We will try and
# learn the inverse. The function will wavy, and thus the inverse will be one to
# many. Thus, simply learning the mapping based on one input and one output is
# impossible. Instead, we will incorporate a few adjacent inputs, which
# collectively should have enough information to identify the correct output.

# Make this reproducible.
set.seed(1)

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

# The domain of the forward mapping is (0,10)
x_min <- 0
x_max <- 10

# For plotting purposes, we'll use a grid of points.
x_grid <- seq(x_min, x_max, length.out=100)
y_grid <- f(x_grid, weights)
y_max <- max(y_grid)
y_min <- min(y_grid)

# Plot the inverse mapping
pdf(file="figure_1.pdf", height=4, width=4)
par(mar=c(4,4,1,1), mgp=c(2,0.8,0), cex.lab=1, cex.axis=0.8)
plot(y_grid, x_grid, type="l", xlab="input", ylab="output")
abline(v=-3)
dev.off()

# Sample x and then derive other features as a time-shift. Finally combine into
# one matrix.
n <- 1000
x1 <- runif(n, min=2, max=10)
x2 <- x1 - 1
x3 <- x1 - 2
x <- cbind(x1, x2, x3)

noise_sd <- 0.4
y_true <- apply(x, 2, f, weights)
y_observed <- y_true + rnorm(3*n, mean=0, sd=noise_sd)

# Make a data frame, swapping "x" and "y"
d <- as.data.frame(cbind(y_observed, x1))
names(d) <- c(paste("x", 1:3, sep=""), "y")
write.table(d, file="data.csv", sep=",", col.names=TRUE,
  row.names=FALSE, quote=FALSE)

# Illustrate how sequential information can help us solve the multi-map problem
example_x <- 6.61029
example_3x <- example_x - 0:2
example_y <- as.numeric(f(example_3x))
example_y_observed <- example_y + rnorm(3, mean=0, sd=noise_sd)

pdf(file="figure_2.pdf", height=4, width=4)
par(mar=c(4,4,1,2), mgp=c(2,0.8,0), cex.lab=1, cex.axis=0.8, xpd=NA)
plot(y_grid, x_grid, type="l", xlab="input", ylab="output")
segments(y_min, example_x, y_max, example_x, col="firebrick", lty=3)
points(example_y_observed, example_3x)
text(y_max+1.0, example_x, labels="z", cex=0.8, col="firebrick")
points(example_y_observed, rep(0,3), pch=17, col="steelblue")
text(example_y_observed, rep(0.5,3), labels=as.character(1:3), cex=0.6, col="steelblue")
text(example_y_observed+1.0, example_3x, col="steelblue")
par(xpd=FALSE)
abline(v=-3, col="gray60")
dev.off()

# END
