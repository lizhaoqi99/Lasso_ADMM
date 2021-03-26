source(file="./admm_lasso.R")

# load data
data <- as.matrix(read.csv("./diabetes.csv", header = TRUE))
X <- data[,1:(ncol(data)-1)]
Y <- data[,ncol(data)]

# run ADMM and calculate time to reach convergence
start = Sys.time()
res <- ADMM_lasso(X,Y,lambda=0.01,rho=5)
end = Sys.time()
print(end-start)
print(paste(res$iter, " iterations", sep=''))
# number of nonzero coefficients
print(paste((sum(res$X.hat != 0)), " nonzero coefficients", sep=''))


# plot
par(mfrow=c(1,3))
plot(res$objective~seq(1,res$iter,by=1), type="l", xlab="iteration", ylab="objective",
     main= "objective function")
plot(res$r.norms~seq(1,res$iter,by=1), type="l", xlab="iteration", ylab="primal residual",
     main= "primal residual")
plot(res$s.norms~seq(1,res$iter,by=1), type="l", xlab="iteration", ylab="dual residual",
     main= "dual residual")