test_lambda_selection <- function() {
  p <- c(rep(8, 4), 2)
  inds <- matrix(c(1, 2, 3, 1, 4, 4, 4, 5), 4, 2)
  v <- list()
  for (i in 1:4) {
    x <- matrix(rnorm(p[i]^2), p[i], p[i])
    v[[i]] <- svd(x)$u[, 1:4]
  }
  x <- matrix(rnorm(p[5]^2), p[5], p[5])
  v[[5]] <- cbind(svd(x)$u[, 1:2], matrix(0, 2, 2))
  X <- list()
  X[[1]] <- v[[1]][, 1:2] %*% t(v[[4]][, 1:2])
  X[[2]] <- v[[2]][, 1:2] %*% t(v[[4]][, c(1, 3)])
  X[[3]] <- v[[3]][, 1:2] %*% t(v[[4]][, 3:4])
  X[[4]] <- v[[1]][, c(1, 3)] %*% t(v[[5]][, 1:2])
  X[[1]][matrix(runif(8 * 8), 8, 8) < 0.2] <- NA

  mmpca::mmpca(
    X,
    inds,
    3,
    parallel = FALSE,
    lambda = c(0.1, 0.1, 0.1),
    max_iter = 20
  )

  # only checking for errors, not checking the result
  TRUE
}

# Use two cores for parallel execution
opt <- options(mc.cores = 2)

set.seed(1)
test_lambda_selection()

# Reset options
options(opt)
