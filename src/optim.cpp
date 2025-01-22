#include <RcppEigen.h>
#include <Rcpp/Lightest>

#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_multimin.h>

#include <ctime>
#include <sstream>

#include "fdf.h"

struct parameters {
  std::vector<Eigen::Map<Eigen::MatrixXd>>* x = nullptr;
  std::vector<Eigen::Map<Eigen::MatrixXd>>* masks = nullptr;
  Eigen::VectorXd* lambda = nullptr;
  int k = 0;
  Eigen::MatrixXi* inds = nullptr;
  Eigen::VectorXi* p = nullptr;
  size_t m = 0;
  size_t n = 0;
  size_t len = 0;
  Eigen::MatrixXi* indices = nullptr;
  int n_threads = 0;
  std::vector<std::size_t>* cidx = nullptr;
};

double gsl_obj(const gsl_vector* theta, void* params) {
  parameters p = *static_cast<parameters*>(params);
  return f_obj(theta->data, *p.x, *p.masks, *p.lambda, p.k, *p.inds, *p.p, p.m, p.n,
               *p.cidx);
}

void gsl_d_obj(const gsl_vector* theta, void* params, gsl_vector* grad) {
  parameters p = *static_cast<parameters*>(params);
  d_obj(grad->data, theta->data, *p.x, *p.masks, *p.lambda, p.k, *p.inds, *p.p, p.m, p.n,
        p.len, *p.indices, p.n_threads, *p.cidx);
}

void gsl_fd_obj(const gsl_vector* theta, void* params, double* f, gsl_vector* grad) {
  parameters p = *static_cast<parameters*>(params);
  d_obj(grad->data, theta->data, *p.x, *p.masks, *p.lambda, p.k, *p.inds, *p.p, p.m, p.n,
        p.len, *p.indices, p.n_threads, *p.cidx);
  *f = f_obj(theta->data, *p.x, *p.masks, *p.lambda, p.k, *p.inds, *p.p, p.m, p.n,
             *p.cidx);
}

// Below replaces the following algorithm:
//
// total = 0
// for view = 0 to n - 1:
//   m = min(k, p[view] - 1)
//   for j = 0 to m - 1
//     total += (p[view] - (j + 1))
//
// Unroll inner loop:
//   total += (p[view] - 1)
//   total += (p[view] - 2)
//   total += (p[view] - 3)
//   ...
//   total += (p[view] - m)
// In one step:
//   total += m * (p[view] - (m + 1) / 2)
int prep_indices_len(const int k, const Eigen::VectorXi& p) {
  const Eigen::ArrayXi m = (p.array() - 1).min(k);
  return (m * p.array() - m * (m + 1) / 2).sum();
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
    cidx[i] = k * p[i - 1] + cidx[i - 1];
  }

  return cidx;
}

template <typename T>
std::vector<T> list_to_vec(Rcpp::List list) {
  std::vector<T> vec;
  vec.reserve(list.length());
  std::transform(list.begin(), list.end(), std::back_inserter(vec), Rcpp::as<T>);

  return vec;
}

// [[Rcpp::export]]
double c_objective(Eigen::Map<Eigen::MatrixXd> theta, Rcpp::List x_list,
                   Rcpp::List masks_list, Eigen::MatrixXi inds, int k, Eigen::VectorXi p,
                   Eigen::VectorXd lambda) {
  if (lambda.size() < 4) {
    const auto i = lambda.size();
    lambda.conservativeResize(4);
    for (auto j = i; j < 4; j++) {
      lambda[j] = 0.0;
    }
  }

  // Convert to 0-based index
  inds.array() -= 1;

  const auto x = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(x_list);
  const auto masks = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(masks_list);

  const auto cidx = compute_cidx(k, p);

  return f_obj(theta.data(), x, masks, lambda, k, inds, p, x.size(), p.size(), cidx);
}

// [[Rcpp::export]]
Eigen::MatrixXd c_grad(Eigen::Map<Eigen::MatrixXd> theta, Rcpp::List x_list,
                       Rcpp::List masks_list, Eigen::MatrixXi inds, int k,
                       Eigen::VectorXi p, Eigen::VectorXd lambda, int n_threads) {
  if (lambda.size() < 4) {
    const auto i = lambda.size();
    lambda.conservativeResize(4);
    for (auto j = i; j < 4; j++) {
      lambda[j] = 0.0;
    }
  }

  const auto x = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(x_list);
  const auto masks = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(masks_list);

  const auto indices_len = prep_indices_len(k, p);
  const auto indices = prep_indices(indices_len, k, p);
  const auto cidx = compute_cidx(k, p);

  Eigen::MatrixXd grad(theta.rows(), theta.cols());
  d_obj(grad.data(), theta.data(), x, masks, lambda, k, inds.array() - 1, p, x.size(),
        p.size(), theta.size(), indices, n_threads, cidx);

  return grad;
}

// [[Rcpp::export]]
Eigen::MatrixXd c_Vxi(Eigen::Map<Eigen::MatrixXd> xi) {
  Eigen::MatrixXd u(xi.rows(), xi.cols());
  f_vxi(u.data(), xi.data(), xi.rows(), xi.cols());

  return u;
}

// [[Rcpp::export]]
Rcpp::List c_optim_mmpca(Eigen::Map<Eigen::MatrixXd> start, Rcpp::List x_list,
                         Rcpp::List masks_list, Eigen::MatrixXi inds, int k,
                         Eigen::VectorXi p, Eigen::VectorXd lambda, int max_iter,
                         bool trace, int n_threads) {
  if (lambda.size() < 4) {
    const auto i = lambda.size();
    lambda.conservativeResize(4);
    for (auto j = i; j < 4; j++) {
      lambda[j] = 0.0;
    }
  }

  // Convert to zero-based indices
  inds.array() -= 1;

  // Convert to vector
  auto x = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(x_list);
  auto masks = list_to_vec<Eigen::Map<Eigen::MatrixXd>>(masks_list);

  Eigen::MatrixXd theta = Eigen::MatrixXd::Zero(start.rows(), start.cols());
  const int indices_len = prep_indices_len(k, p);
  Eigen::MatrixXi indices = prep_indices(indices_len, k, p);
  std::vector<std::size_t> cidx = compute_cidx(k, p);

  parameters params;
  params.x = &x;
  params.masks = &masks;
  params.lambda = &lambda;
  params.k = k;
  params.inds = &inds;
  params.p = &p;
  params.m = x.size();
  params.n = p.size();
  params.len = theta.size();
  params.indices = &indices;
  params.n_threads = n_threads;
  params.cidx = &cidx;

  gsl_multimin_function_fdf min_func;
  min_func.f = gsl_obj;
  min_func.df = gsl_d_obj;
  min_func.fdf = gsl_fd_obj;
  min_func.n = theta.size();
  min_func.params = static_cast<void*>(&params);

  gsl_vector_const_view gsl_start =
      gsl_vector_const_view_array(start.data(), start.size());

  gsl_multimin_fdfminimizer* s = gsl_multimin_fdfminimizer_alloc(
      gsl_multimin_fdfminimizer_vector_bfgs2, theta.size());
  gsl_multimin_fdfminimizer_set(s, &min_func, &gsl_start.vector, 0.01, 1);

  int ret = 0;
  int it = 0;
  for (; it < max_iter; it++) {
    ret = gsl_multimin_fdfminimizer_iterate(s);
    if (trace && it % 10 == 0) {
      time_t t;
      time(&t);

      // clang-format off
      std::stringstream ss;
      ss << std::setw(19) << std::string(ctime(&t)).substr(0, 19)
         << "\t" << it
         << "\t" << std::setprecision(6) << std::fixed << s->f
         << "\t" << std::setprecision(6) << std::fixed << gsl_blas_dnrm2(gsl_multimin_fdfminimizer_dx(s))
         << "\t" << std::setprecision(6) << std::fixed << gsl_blas_dnrm2(s->gradient)
         << "\r";
      Rcpp::Rcout << ss.str();
      // clang-format on

      if (it % 1000 == 0) Rcpp::Rcout << std::endl;
    }
    // Stop if non-zero return value
    if (ret) break;

    Rcpp::checkUserInterrupt();
  }
  double step_size = gsl_blas_dnrm2(gsl_multimin_fdfminimizer_dx(s));

  std::string message;
  int status = 0;
  if (ret) {
    message = std::string(gsl_strerror(ret));
    status = 1;
  } else {
    message = std::string("maximum iterations reached");
    status = 2;
  }

  if (trace && it % 10 != 0) {
    time_t t;
    time(&t);

    // clang-format off
    std::stringstream ss;
    ss << std::setw(19) << std::string(ctime(&t)).substr(0, 19)
       << "\t" << it
       << "\t" << std::setprecision(6) << std::fixed << s->f
       << "\t" << std::setprecision(6) << std::fixed << gsl_blas_dnrm2(gsl_multimin_fdfminimizer_dx(s))
       << "\t" << std::setprecision(6) << std::fixed << gsl_blas_dnrm2(s->gradient)
       << std::endl;
    Rcpp::Rcout << ss.str();
    //clang-format on
  }
  if (trace && it % 10 == 0 && it % 1000 >= 10) Rcpp::Rcout << std::endl;

  gsl_vector_view gsl_opt = gsl_vector_view_array(theta.data(), theta.size());
  gsl_vector_memcpy(&gsl_opt.vector, s->x);

  gsl_multimin_fdfminimizer_free(s);

  // Compute unpenalized objective value
  Eigen::VectorXd lambda_zero = Eigen::VectorXd::Zero(4);
  parameters up_params;
  up_params.x = &x;
  up_params.masks = &masks;
  up_params.lambda = &lambda_zero;
  up_params.k = k;
  up_params.inds = &inds;
  up_params.p = &p;
  up_params.m = x.size();
  up_params.n = p.size();
  up_params.len = theta.size();
  up_params.indices = &indices;
  up_params.n_threads = n_threads;
  up_params.cidx = &cidx;

  // gsl_vector_const_view gsl_opt = gsl_vector_const_view_array(theta.data(), theta.size());
  double unpenalized_loss = gsl_obj(&gsl_opt.vector, static_cast<void*>(&up_params));
  if (trace) {
    time_t t;
    time(&t);

    // clang-format off
    std::stringstream ss;
    ss << std::setw(19) << std::string(ctime(&t)).substr(0, 19)
       << "\t" << "Unpenalized loss:"
       << "\t" << std::setprecision(6) << std::fixed << unpenalized_loss
       << std::endl;
    Rcpp::Rcout << ss.str();
    // clang-format on
  }

  Rcpp::List res;
  res["theta"] = theta;
  res["iter"] = it;
  res["status"] = status;
  res["upval"] = unpenalized_loss;
  res["stepsize"] = step_size;
  res["msg"] = message;

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
