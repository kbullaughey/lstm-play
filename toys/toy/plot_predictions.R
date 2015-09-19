# This script expects two arguments, the file containing predictions and the figure name
args <- commandArgs(trailingOnly = TRUE)
stopifnot(length(args) == 2)
predictions_file = args[1]
figure_file = args[2]

f <- function(x) 2 + 0.5*sin(x/2 - 1) + sin(x) + sin(x*2 + 2) + sin(x/4 + 1)
pred = read.table(predictions_file)$V1

x_grid <- seq(0, 10, length=length(pred))
y_grid <- f(x_grid)

#---------
# Figure 5
#---------

# Plot the inverse mapping
pdf(file=figure_file, height=4, width=4)
par(mar=c(4,4,1,1), mgp=c(2,0.8,0), cex.lab=1, cex.axis=0.8)
plot(range(y_grid), range(x_grid), type="n", xlab="input", ylab="output")
lines(y_grid, pred, col="orange", lwd=2)
lines(y_grid, x_grid, col="black")
dev.off()
