f1 <- function(x) {
  sin(x*3)
}
f2 <- function(x) {
  (x-5)^2
}
f3 <- function(x) {
  log(x+0.1)
}
f4 <- function(x) {
  1+3*x
}
f5 <- function(x) {
  (x/2-4)^2
}

weights <- c(1,0.2,-1,-0.1,-0.2)

f <- function(x, w=weights) {
  z = cbind(f1(x), f2(x), f3(x), f4(x), f5(x))
  (z %*% w) * (1 + 0.2*sin(1.5*x-1))
}

# Parameters
high_end = 11
num_inputs = 4
n <- 5000

# We'll try and learn a function inverse
x_grid <- seq(0, high_end, length.out=100)
y_grid <- f(x_grid, weights)

# Sample x and then derive other features as a time-shift. Finally combine into
# one matrix.
x1 <- runif(n, min=num_inputs-1, max=high_end)
x234 <- sapply(1:3, function(i) x1-i)
x <- cbind(x1, x234)

y_true <- apply(x, 2, f, weights)
y_observed <- y_true + rnorm(num_inputs*n, mean=0, sd=0.3)

y_range <- range(c(y_true, y_grid, y_observed))

# Make a data frame, swapping "x" and "y"
d <- as.data.frame(cbind(y_observed, x1))
ytrue <- as.data.frame(y_true)
names(d) <- c(paste("x", 1:num_inputs, sep=""), "y")
names(ytrue) <- paste("x", 1:num_inputs, sep="")

# Randomly pick 2, 3, or 4 inputs (always starting from x1). This way we'll
# be modeling variable-length sequences.
fout <- file('noisy_inputs.txt', 'w')
sequences <- lapply(1:n, function(i) {
  len <- ceiling(runif(1,min=1, max=num_inputs))
  cat(paste(d[i,1:len], collapse=","), "\n", file=fout, sep="")
})
close(fout)
fout <- file('true_inputs.txt', 'w')
sequences <- lapply(1:n, function(i) {
  len <- ceiling(runif(1,min=1, max=num_inputs))
  cat(paste(ytrue[i,1:len], collapse=","), "\n", file=fout, sep="")
})
close(fout)
write.table(d$y, file='outputs.txt', quote=FALSE, col.names=FALSE,
  row.names=FALSE)

# Visualization code
pdf(file="sim.pdf", height=4, width=4)
palette(c("gray80", "black", "firebrick", "olivedrab", "plum", "salmon", "steelblue"))
par(mar=c(4,4,1,1), mgp=c(2,0.8,0), cex.lab=1, cex.axis=0.8)
plot(y_range, c(0,high_end), type="n", xlab="y", ylab="f-1(y)")
points(y_observed, x, col=1, cex=0.8)
lines(y_grid, x_grid, col=2, lwd=1.5)
# For a sampling, plot the three points linked together
trash <- lapply(1:4, function(i) {
  lines(d[i,1:num_inputs], x[i,], col=i+3, lwd=2)
})
dev.off()


# END
