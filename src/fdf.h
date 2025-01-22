#ifndef __FDF_H_
#define __FDF_H_

#include <vector>
#include <Eigen/Core>

void f_vxi(double * v, const double * x, const int p, const int k);
double f_obj(const double* theta, const std::vector<Eigen::Map<Eigen::MatrixXd>>& x,
             const std::vector<Eigen::Map<Eigen::MatrixXd>>& masks,
             const Eigen::VectorXd& lambda, const int k, const Eigen::MatrixXi& inds,
             const Eigen::VectorXi& p, const int m, const int n, const std::vector<std::size_t>& cidx);
void d_obj(double* grad, const double* theta, const std::vector<Eigen::Map<Eigen::MatrixXd>>& x,
           const std::vector<Eigen::Map<Eigen::MatrixXd>>& masks, const Eigen::VectorXd& lambda, const int k,
           const Eigen::MatrixXi& inds, const Eigen::VectorXi& p, const int m, const int n,
           const int len, const Eigen::MatrixXi& indices, const int n_threads,
           const std::vector<std::size_t>& cidx);
void inv_v(double * xi, double * c_t, int n);

#endif // __FDF_H_