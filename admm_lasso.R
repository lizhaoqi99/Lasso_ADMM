### Inspired by Dr. Boyd's MATLAB program                                      
### which can be found at https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html

# helper functions
L1norm <- function(x) {
  return(sum(abs(x)))
}

L2norm <- function(x) {
  return(sqrt(sum(x^2)))
}

soft_thres <- function(x, lambda) {
  prox <- sign(x) * pmax(rep(0, length(x)), abs(x)-lambda)
  return(prox)
}

objective <- function(A, b, x, lambda) {
  m <- A %*% x - b
  p <- 1/2 * t(m) %*% m + lambda * L1norm(x)
  return(p)
}

# main function
# background knowledge from text p.43
# text: https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
ADMM_lasso <- function(A,b,lambda,rho,iter=1000,e.abs=1E-4,e.rel=1E-2){
  # A: design matrix
  # b: response vector
  # lambda: penalty parameter for the primal problem
  # rho: penalty parameter for the augmented Lagrangian
  # iter: max number of iterations
  # e.abs: absolute tolerance stopping constant
  # e.rel: relative tolerance stopping constant
  
  n <- nrow(A) 
  p <- ncol(A)
  
  # initialize coefficient matrix X 
  X <- matrix(0, nrow=iter, ncol=p) 
  X[1,] <- rep(0, p)
  
  # initialize Z and U matrices
  Z <- matrix(0, nrow=iter, ncol=p)  
  U <- rep(0, p)   
  
  # initialize objective function
  obj <- rep(0, iter)
  obj[1] <- objective(A, b, X[1,], lambda)
  # obj[1] <- 0
  
  # compute (AtA+pI)^-1 (for x update) which is fixed
  AtA <- t(A) %*% A
  inv_m <- solve(AtA + diag(rho, p))
  
  # initialize residuals
  r <- X[1,]-Z[1,]    # primal residual
  #r <- 0
  s <- 0    # dual residual
  
  # initialize iteration count
  k <- 0
  
  r.norms <- rep(0, iter)
  s.norms <- rep(0, iter)
  r.norms[1] <- L2norm(r)
  s.norms[1] <- L2norm(s)
  
  # ADMM updates
  for (k in 2:iter){
    # update X, Z, U, and objectives
    Atb <- t(A) %*% b 
    X[k,] <- inv_m %*% (Atb + rho * (Z[k-1,]-U))
    Z[k,] <- soft_thres(X[k,] + U, lambda/rho) # z update
    U <- U + X[k,] - Z[k,] # dual update
    obj[k] <- objective(A, b, X[k,], lambda) # update objective
    
    # calculate residuals for iteration k
    r <- X[k,] - Z[k,] # update primal residual
    s <- -rho * (Z[k,] - Z[k-1,]) # update dual residual (text p.18)
    
    # compute L2-norm for both residuals
    r.norm <- L2norm(r)
    r.norms[k] <- r.norm
    s.norm <- L2norm(s)
    s.norms[k] <- s.norm
    
    # more info can be found on text p.19:
    # feasibility tolerance for primal feasibility condition
    e.primal <- sqrt(p) * e.abs + e.rel * max(L2norm(X[k,]), L2norm(Z[k,])) 
    # feasibility tolerance for dual feasibility condition
    e.dual <-  sqrt(n) * e.abs + e.rel * L2norm(U)
    
    # check termination conditions
    if (r.norm <= e.primal && s.norm <= e.dual){
      # remove excess X and objective
      X <- X[-((k+1):nrow(X)),]
      obj <- obj[-((k+1):length(obj))]
      r.norms <- r.norms[-((k+1):length(r.norms))]
      s.norms <- s.norms[-((k+1):length(s.norms))]
      break
    }
  }
  res <- list("X.hat"=X[nrow(X),], "X"=X, "objective"=obj, "iter"=k,
              "r.norms"=r.norms,"s.norms"=s.norms)
  return(res)
}