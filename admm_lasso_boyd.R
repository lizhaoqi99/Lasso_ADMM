# example from https://web.stanford.edu/~boyd/papers/admm/lasso/lasso_example.html

source(file="./admm_lasso.R")
library("Matrix")

# import data
m <- 500 # number of samples
n <- 1000 # number of features
p <- 0.1 # sparsity density: percentage of non-zero elements

set.seed(0)
# generate a random sparse matrix
x0 <- matrix(rsparsematrix(n,1,p))
A <- matrix(rnorm(m*n),nrow=m)

# normalize columns
for (i in 1:ncol(A)){
  A[,i] <- A[,i]/sqrt(sum(A[,i]*A[,i]))
}

b <- A%*%x0 + sqrt(0.001)*matrix(rnorm(m))

# set regularization term lambda
lambda <- 0.1*base::norm(t(A)%*%b, "F")

rho <- 1

# start <- Sys.time()
# res <- ADMM_lasso(A,b,lambda,rho)
# end <- Sys.time()
# print(end-start)
# print(paste(res$iter, " iterations", sep=''))
# # number of nonzero coefficients
# print(paste((sum(res$X.hat != 0)), " nonzero coefficients", sep=''))
# 
# # plot
# par(mfrow=c(1,3))
# plot(res$objective~seq(1,res$iter,by=1), type="l", xlab="iteration", ylab="objective",
#      main= "objective function")
# plot(res$r.norms~seq(1,res$iter,by=1), type="l", xlab="iteration", ylab="primal residual",
#     main= "primal residual")
# plot(res$s.norms~seq(1,res$iter,by=1), type="l", xlab="iteration", ylab="dual residual",
#      main= "dual residual")


library("ADMM")
start <- Sys.time()
output <- admm.lasso(A, b, lambda)
end <- Sys.time()
print(end-start)

niter   = length(output$history$s_norm)
history = output$history
opar <- par(no.readonly=TRUE)
par(mfrow=c(1,3))
plot(1:niter, history$objval, "b", main="cost function")
plot(1:niter, history$r_norm, "b", main="primal residual")
plot(1:niter, history$s_norm, "b", main="dual residual")
par(opar)