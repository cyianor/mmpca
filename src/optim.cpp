#include <RcppEigen.h>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>

#include <time.h>

#include "fdf.h"

double dx_nrm2(gsl_multimin_fdfminimizer* s) {
  return gsl_blas_dnrm2(gsl_multimin_fdfminimizer_dx(s));
}

void minimize(double* theta, int* iter, int* status, char* msg, double* stepsize,
              void* params, int len, gsl_multimin_function_fdf* gsl_multimin_params,
              const double* start, const bool trace) {
  gsl_vector_view gsl_opt = gsl_vector_view_array(theta, len);
  gsl_vector_const_view gsl_start = gsl_vector_const_view_array(start, len);

  gsl_multimin_fdfminimizer* s = gsl_multimin_fdfminimizer_alloc(
      gsl_multimin_fdfminimizer_vector_bfgs2, gsl_multimin_params->n);
  gsl_multimin_fdfminimizer_set(s, gsl_multimin_params, &gsl_start.vector, 0.01, 1);

  int status_iter;
  for (*iter = 0; *iter < 20001; (*iter)++) {
    status_iter = gsl_multimin_fdfminimizer_iterate(s);
    if (trace && *iter % 10 == 0) {
      time_t t;
      time(&t);
      Rprintf("%.19s\t%d\t%f\t%f\t%f\r", ctime(&t), *iter, s->f, dx_nrm2(s),
              gsl_blas_dnrm2(s->gradient));
      R_FlushConsole();
      if (*iter % 1000 == 0) Rprintf("\n");
    }
    if (status_iter) break;

    R_CheckUserInterrupt();
  }
  *stepsize = dx_nrm2(s);

  if (status_iter) {
    strcpy(msg, gsl_strerror(status_iter));
    *status = 1;
  } else {
    strcpy(msg, "maximum iterations reached");
    *status = 2;
  }

  if (trace && *iter % 10 != 0) {
    time_t t;
    time(&t);
    Rprintf("%.19s\t%d\t%f\t%f\t%f\n", ctime(&t), *iter, s->f, dx_nrm2(s),
            gsl_blas_dnrm2(s->gradient));
    R_FlushConsole();
  }
  if (trace && *iter % 10 == 0 && *iter % 1000 >= 10) Rprintf("\n");

  gsl_vector_memcpy(&gsl_opt.vector, s->x);

  gsl_multimin_fdfminimizer_free(s);
}

struct parameters {
  const std::vector<Eigen::Map<Eigen::MatrixXd>>* x;
  const std::vector<Eigen::Map<Eigen::MatrixXd>>* masks;
  const Eigen::VectorXd* lambda;
  const int k;
  const Eigen::MatrixXi* inds;
  const Eigen::VectorXi* p;
  const size_t m, n, len;
  const Eigen::MatrixXi* indices;
  const int num_threads;
  const std::vector<std::size_t>* cidx;
};

double gsl_obj(const gsl_vector* theta, void* params) {
  parameters p = *static_cast<parameters*>(params);
  return f_obj(theta->data, *p.x, *p.masks, *p.lambda, p.k, *p.inds, *p.p, p.m, p.n,
               *p.cidx);
}

void gsl_d_obj(const gsl_vector* theta, void* params, gsl_vector* grad) {
  parameters p = *static_cast<parameters*>(params);
  d_obj(grad->data, theta->data, *p.x, *p.masks, *p.lambda, p.k, *p.inds, *p.p, p.m, p.n,
        p.len, *p.indices, p.num_threads, *p.cidx);
}

void gsl_fd_obj(const gsl_vector* theta, void* params, double* f, gsl_vector* grad) {
  parameters p = *static_cast<parameters*>(params);
  d_obj(grad->data, theta->data, *p.x, *p.masks, *p.lambda, p.k, *p.inds, *p.p, p.m, p.n,
        p.len, *p.indices, p.num_threads, *p.cidx);
  *f = f_obj(theta->data, *p.x, *p.masks, *p.lambda, p.k, *p.inds, *p.p, p.m, p.n,
             *p.cidx);
}

int prep_indices_len(const int k, const Eigen::VectorXi& p) {
  int indices_len = 0;
  for (Eigen::Index view = 0, n = p.size(); view < n; view++) {
    for (int j = 0; j < k && j < p[view] - 1; j++) {
      for (int i = j + 1; i < p[view]; i++) {
        indices_len++;
      }
    }
  }
  return indices_len;
}

Eigen::MatrixXi prep_indices(const int indices_len, const int k,
                             const Eigen::VectorXi& p) {
  Eigen::MatrixXi indices(3, indices_len);
  Eigen::Index l = 0;
  for (Eigen::Index view = 0, n = p.size(); view < n; view++) {
    for (int j = 0; j < k && j < p[view] - 1; j++) {
      for (int i = j + 1; i < p[view]; i++) {
        indices(0, l) = view;
        indices(1, l) = i;
        indices(2, l) = j;

        l++;
      }
    }
  }

  return indices;
}

static std::vector<std::size_t> compute_cidx(const int k, const Eigen::VectorXi& p) {
  std::vector<std::size_t> cidx(p.size() + 1);
  cidx[0] = 0;
  for (size_t i = 1, n = cidx.size(); i < n; i++) {
    cidx[i] += k * p[i - 1];
  }

  return cidx;
}

void optim(double* theta, const double* start, size_t len,
           const std::vector<Eigen::Map<Eigen::MatrixXd>>& x,
           const std::vector<Eigen::Map<Eigen::MatrixXd>>& masks,
           const Eigen::MatrixXi& inds, const int k, size_t m, const Eigen::VectorXi& p,
           const Eigen::VectorXd& lambda, int* iter, int* status, char* msg,
           double* upval, double* stepsize, const bool trace, const int num_threads) {
  const int indices_len = prep_indices_len(k, p);
  const Eigen::MatrixXi indices = prep_indices(indices_len, k, p);
  std::vector<std::size_t> cidx = compute_cidx(k, p);

  parameters params{&x,    &masks,   &lambda,     k,
                    &inds, &p,       m,           static_cast<size_t>(p.size()),
                    len,   &indices, num_threads, &cidx};

  gsl_multimin_function_fdf gsl_multimin_params{gsl_obj, gsl_d_obj, gsl_fd_obj,
                                                params.len, static_cast<void*>(&params)};
  minimize(theta, iter, status, msg, stepsize, &params, params.len, &gsl_multimin_params,
           start, trace);
  Eigen::VectorXd lambda_zero = Eigen::VectorXd::Zero(4);
  parameters up_params{
      &x,  &masks,   &lambda_zero, k,    &inds, &p, m, static_cast<size_t>(p.size()),
      len, &indices, num_threads,  &cidx};
  gsl_vector_const_view gsl_opt = gsl_vector_const_view_array(theta, params.len);
  *upval = gsl_obj(&gsl_opt.vector, &up_params);
  if (trace) {
    time_t t;
    time(&t);
    Rprintf("%.19s\tUnpenalized loss:\t%f\n", ctime(&t), *upval);
    R_FlushConsole();
  }
}

template <typename T>
std::vector<T> list_to_vec(Rcpp::List list) {
	std::vector<T> vec;
	vec.reserve(list.length());
	std::transform(list.begin(), list.end(), std::back_inserter(vec), Rcpp::as<T>);

	return vec;
}

// [[Rcpp::export]]
double c_objective(Eigen::Map<Eigen::MatrixXd> theta, Rcpp::List x, Rcpp::List masks,
             Eigen::MatrixXi inds, int k, Eigen::VectorXi p, Eigen::VectorXd lambda) {
  if (lambda.size() < 4) {
    const auto i = lambda.size();
    lambda.resize(4);
    for (auto j = i; j < 4; j++) {
      lambda[j] = 0.0;
    }
  }

  std::vector<Eigen::Map<Eigen::MatrixXd>> c_x = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(x);
  std::vector<Eigen::Map<Eigen::MatrixXd>> c_masks = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(x);

  std::vector<std::size_t> cidx = compute_cidx(k, p);

  return f_obj(theta.data(), c_x, c_masks, lambda, k, inds.array() - 1, p, x.size(), p.size(), cidx);
}

// [[Rcpp::export]]
Eigen::MatrixXd c_grad(Eigen::Map<Eigen::MatrixXd> theta, Rcpp::List x, Rcpp::List masks,
                       Eigen::MatrixXi inds, int k, Eigen::VectorXi p,
                       Eigen::VectorXd lambda, int num_threads) {
  if (lambda.size() < 4) {
    const auto i = lambda.size();
    lambda.resize(4);
    for (auto j = i; j < 4; j++) {
      lambda[j] = 0.0;
    }
  }

  std::vector<Eigen::Map<Eigen::MatrixXd>> c_x = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(x);
  std::vector<Eigen::Map<Eigen::MatrixXd>> c_masks = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(x);

  const int indices_len = prep_indices_len(k, p);
  const Eigen::MatrixXi indices = prep_indices(indices_len, k, p);
  std::vector<std::size_t> cidx = compute_cidx(k, p);

  Eigen::MatrixXd grad(theta.rows(), theta.cols());
  d_obj(grad.data(), theta.data(), c_x, c_masks, lambda, k, inds.array() - 1, p, x.size(), p.size(),
        theta.size(), indices, num_threads, cidx);

  return grad;
}

// [[Rcpp::export]]
Eigen::MatrixXd c_Vxi(Eigen::Map<Eigen::MatrixXd> xi) {
  Eigen::MatrixXd u(xi.rows(), xi.cols());
  f_vxi(u.data(), xi.data(), xi.rows(), xi.cols());

  return u;
}

// [[Rcpp::export]]
Rcpp::List c_optim_mmpca(Eigen::Map<Eigen::MatrixXd> start, Rcpp::List x,
                         Rcpp::List masks, Eigen::MatrixXi inds, int k, Eigen::VectorXi p,
                         Eigen::VectorXd lambda, bool trace, int num_threads) {
  if (lambda.size() < 4) {
    const auto i = lambda.size();
    lambda.resize(4);
    for (auto j = i; j < 4; j++) {
      lambda[j] = 0.0;
    }
  }

  std::vector<Eigen::Map<Eigen::MatrixXd>> c_x = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(x);
  std::vector<Eigen::Map<Eigen::MatrixXd>> c_masks = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(x);

  Eigen::MatrixXd theta(start.rows(), start.cols());
  int iter, status;
  double upval, stepsize;
  char msg[100];
  optim(theta.data(), start.data(), theta.size(), c_x, c_masks, inds.array() - 1, k, x.size(), p,
        lambda, &iter, &status, msg, &upval, &stepsize, trace, num_threads);

  Rcpp::List res(6);
  res[0] = theta;
  res[1] = iter;
  res[2] = status;
  res[3] = upval;
  res[4] = stepsize;
  res[5] = msg;

  return res;
}

// [[Rcpp::export]]
Eigen::MatrixXd c_invVinner(Eigen::Map<Eigen::MatrixXd> t) {
  const auto n = t.rows();

  Eigen::MatrixXd xi(n, n);
  inv_v(xi.data(), t.data(), static_cast<int>(n));

  return -xi.transpose();
}

// [[Rcpp::export]]
void c_init_parallel() { Eigen::initParallel(); }
