d = read.table(file='bench_linear_parallel.out', stringsAsFactors=FALSE, sep=",")
names(d) <- c("M", "maps", "reps", "mode", "time")

pdf(file="gpu_elapsed_time_and_size.pdf", height=4, width=5)
par(mar=c(4,4,1,1), mgp=c(2,0.6,0.2), cex.lab=1, cex.axis=0.6)
palette(c("firebrick", "steelblue", "brown"))
plot(range(d$M), range(d$time), type="n", xlab="M, MxM matrix", ylab="elapsed time", axes=FALSE)
axis(1, at=sort(unique(d$M)))
axis(2)
mode <- c("cpu", "gpu")
trash <- lapply(1:length(uniqMaps), function(i) {
  sel <- d$mode == mode[i] & d$maps == 10
  lines(d$M[sel], d$time[sel], col=i)
  points(d$M[sel], d$time[sel], col=i, cex=0.8)
})
legend("topleft", inset=0.1, legend=mode, col=1:2, lwd=1, pch=1)
dev.off()
