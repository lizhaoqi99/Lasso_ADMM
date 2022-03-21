library("Matrix")

# import data
m <- 1000 # number of samples
n <- 10 # number of features
p <- 0.1 # sparsity density: percentage of non-zero elements

set.seed(0)
# generate a random sparse matrix
x0 <- matrix(Matrix::rsparsematrix(n,1,p))
A <- matrix(rnorm(m*n),nrow=m)
# normalize columns
for (i in 1:ncol(A)){
  A[,i] <- A[,i]/sqrt(sum(A[,i]*A[,i]))
}

b <- A%*%x0 + sqrt(0.001)*matrix(rnorm(m))



#------------------------------------------------------#


# 10-fold Cross-validation to choose the best tuning parameter lambda for glmnet
set.seed(123)
train <- sample(1:nrow(A), floor(0.6*nrow(A))) # use 60% of the samples as training set 

# 10-fold cross-validation
set.seed(123)
cv <- cv.glmnet(A[train, ],b[train],alpha =1)
plot(cv)
lambda <- cv$lambda.min

# compare different rhos (rho: penalty parameter for the augmented Lagrangian)
rho <- c(0.5, 1.5, 5, 10, 20)
res <- c()
time <- c()
objs <- c()
iters <- c()
r.norms <- c()
s.norms <- c()

objs2 <- c()
iters2 <- c()
for(i in 1:length(rho)) {
  start <- Sys.time()
  res <- ADMM_lasso(A,b,lambda,rho[i])
  end <- Sys.time()
  time <- c(time,end-start)
  
  objs <- c(objs,list(res$objective))
  iters <- c(iters,res$iter)
  r.norms <- c(r.norms,list(res$r.norms))
  s.norms <- c(s.norms,list(res$s.norms))
  
  res2 <- ADMM_lasso(A,b,lambda,rho[i],e.abs=1E-8,e.rel=1E-4)
  objs2 <- c(objs2,list(res2$objective))
  iters2 <- c(iters2,res2$iter)
}


print(cbind(rho,iters,time))


cols <- c("green", "red", "purple", "orange")


# plot objectives
plot(unlist(objs[1]), type="l", xlab="iteration", ylab="objective",
     main= "objective function", xlim=range(0,max(iters)),
     ylim=range(min(unlist(objs)),max(unlist(objs))))
for(i in 2:length(rho)) {
  lines(seq(1,unlist(iters[i]),by=1),unlist(objs[i]),col=cols[i-1])
}
legend("topright",legend=rho,title="Penalty Parameter",col=c("black",cols),pch=19)

# plot objectives
plot(unlist(objs2[1]), type="l", xlab="iteration", ylab="objective",
     main= "objective function", xlim=range(0,max(iters2)),
     ylim=range(min(unlist(objs2)),max(unlist(objs2))))
for(i in 2:length(rho)) {
  lines(seq(1,unlist(iters2[i]),by=1),unlist(objs2[i]),col=cols[i-1])
}
legend("topright",legend=rho,title="Penalty Parameter",col=c("black",cols),pch=19)



# plot primal residuals
plot(unlist(r.norms[1])~seq(1,iters[1],by=1), type="l", xlab="iteration", ylab="primal residual",
     main= "primal residual", xlim=range(1,max(iters)), ylim=range(0,max(unlist(r.norms))))
for(i in 2:5) {
  lines(seq(1,unlist(iters[i]),by=1),unlist(r.norms[i]),col=cols[i-1])
}
legend("topright",legend=rho,title="Penalty Parameter",col=c("black",cols),pch=19)



# plot dual residuals
plot(unlist(s.norms[1])~seq(1,iters[1],by=1), type="l", xlab="iteration", ylab="dual residual",
     main= "dual residual", xlim=range(1,max(iters)), ylim=range(0,max(unlist(s.norms))))
for(i in 2:5) {
  lines(seq(1,unlist(iters[i]),by=1),unlist(s.norms[i]),col=cols[i-1])
}
legend("topright",legend=rho,title="Penalty Parameter",col=c("black",cols),pch=19)