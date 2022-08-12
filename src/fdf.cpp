#include <RcppEigen.h>
#include <gsl/gsl_sf_trig.h>

#include <iostream>
using namespace std;

using pvt = Eigen::Array<Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic>,
                         Eigen::Dynamic, 1>;

// small constant used in soft abs and soft sign
const double eps = 1e-6;
const double inveps = 1.0 / eps;

template <typename Var, typename Const>
void partialvrs(pvt* pvs, Eigen::MatrixBase<Var> const& V_,
                const Eigen::MatrixBase<Const>& xi) {
  Eigen::MatrixBase<Var>& V = const_cast<Eigen::MatrixBase<Var>&>(V_);
  int k = xi.cols();
  int p = xi.rows();
  V.setIdentity();
  (*pvs)(0) = Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic>(p, k);
  (*pvs)(1) = Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic>(p, k);
  for (int j = k - 1; j >= 0; j--) {
    for (int i = p - 1; i >= j + 1; i--) {
      (*pvs)(0)(i, j) = Eigen::MatrixXd(V.row(i));
      (*pvs)(1)(i, j) = Eigen::MatrixXd(V.row(j));
      double cx = cos(xi(i, j)) - 1;
      double sx = sin(xi(i, j));
      V.row(i) += -sx * (*pvs)(1)(i, j) + cx * (*pvs)(0)(i, j);
      V.row(j) += cx * (*pvs)(1)(i, j) + sx * (*pvs)(0)(i, j);
    }
  }
}

template <typename Mat>
void partialvls(pvt* pvs, const Eigen::MatrixBase<Mat>& xi) {
  int k = xi.cols();
  int p = xi.rows();
  Eigen::MatrixXd V(p, p);
  V.setIdentity();
  (*pvs)(2) = Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic>(p, k);
  (*pvs)(3) = Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, Eigen::Dynamic>(p, k);
  for (int j = 0; j < k; j++) {
    for (int i = j + 1; i < p; i++) {
      (*pvs)(2)(i, j) = Eigen::MatrixXd(V.block(0, i, i + 1, 1));
      (*pvs)(3)(i, j) = Eigen::MatrixXd(V.block(0, j, i + 1, 1));
      double cx = cos(xi(i, j)) - 1;
      double sx = sin(xi(i, j));
      V.block(0, i, i + 1, 1) += cx * (*pvs)(2)(i, j) + sx * (*pvs)(3)(i, j);
      V.block(0, j, i + 1, 1) += -sx * (*pvs)(2)(i, j) + cx * (*pvs)(3)(i, j);
    }
  }
}

template <typename Var, typename Const>
void Vxi(Eigen::MatrixBase<Var> const& V_, const Eigen::MatrixBase<Const>& xi) {
  Eigen::MatrixBase<Var>& V = const_cast<Eigen::MatrixBase<Var>&>(V_);
  V.setIdentity();
  int p = V.rows();
  for (int j = xi.cols() - 1; j >= 0; j--) {
    for (int i = p - 1; i >= j + 1; i--) {
      Eigen::MatrixXd vi(V.row(i));
      double cx = cos(xi(i, j)) - 1;
      double sx = sin(xi(i, j));
      V.row(i) += -sx * V.row(j) + cx * vi;
      V.row(j) += cx * V.row(j) + sx * vi;
    }
  }
}

void f_vxi(double* v, const double* xi, int p, int k) {
  Eigen::Map<Eigen::MatrixXd> eig_v(v, p, k);
  Eigen::Map<const Eigen::MatrixXd> eig_x(xi, p, k);
  Vxi(eig_v, eig_x);
}

double f_obj(const double* theta, const std::vector<Eigen::Map<Eigen::MatrixXd>>& x,
             const std::vector<Eigen::Map<Eigen::MatrixXd>>& masks,
             const Eigen::VectorXd& lambda, const int k, const Eigen::MatrixXi& inds,
             const Eigen::VectorXi& p, const int m, const int n,
             const std::vector<std::size_t>& cidx) {
  double loss = 0;

  Eigen::Map<const Eigen::MatrixXd> D(theta + cidx[n], k, n);
  Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, 1> V(n);

  for (int i = 0; i < n; i++) {
    Eigen::Map<const Eigen::MatrixXd> xi(theta + cidx[i], p[i], k);
    V(i) = Eigen::MatrixXd(p[i], k);
    Vxi(V(i), xi);
    Eigen::MatrixXd vd = V(i) * D.col(i).asDiagonal();
    loss += lambda[2] * vd.array().abs().sum();
    for (int j = 0; j < p[i]; j++) {
      loss += lambda[3] * sqrt(vd.row(j).array().square().sum());
    }
  }

  for (int i = 0; i < m; i++) {
    int row = inds(i, 0);
    int col = inds(i, 1);
    loss += (x[i] - V(row) * (D.col(row).cwiseProduct(D.col(col))).asDiagonal() *
                        V(col).transpose())
                .cwiseProduct(masks[i])
                .array()
                .square()
                .sum();
  }

  loss +=
      lambda[0] *
      (eps * (inveps * D.array()).unaryExpr([](double x) { return gsl_sf_lncosh(x); }))
          .abs()
          .sum();

  for (int i = 0; i < k; i++) {
    loss += lambda[1] * sqrt(D.row(i).array().square().sum());
  }

  return loss;
}

void d_obj(double* grad, const double* theta,
           const std::vector<Eigen::Map<Eigen::MatrixXd>>& x,
           const std::vector<Eigen::Map<Eigen::MatrixXd>>& masks,
           const Eigen::VectorXd& lambda, const int k, const Eigen::MatrixXi& inds,
           const Eigen::VectorXi& p, const int m, const int n, const int len,
           const Eigen::MatrixXi& indices, const int num_threads,
           const std::vector<std::size_t>& cidx) {
  memset(grad, 0.0, sizeof(double) * len);

  Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, 1> V(n);
  Eigen::Array<pvt, Eigen::Dynamic, 1> pvss(n);
  for (int i = 0; i < n; i++) {
    pvss(i) = pvt(4);
  }

  // precalculate rotations and partial rotations
  for (int j = 0; j < 2 * n; j++) {
    int i = j / 2;
    Eigen::Map<const Eigen::MatrixXd> xi(theta + cidx[i], p[i], k);
    if (j % 2 == 0) {
      V(i) = Eigen::MatrixXd(p[i], k);
      partialvrs(&pvss(i), V(i), xi);
    } else {
      partialvls(&pvss(i), xi);
    }
  }

  Eigen::Map<const Eigen::MatrixXd> D(theta + cidx[n], k, n);
  Eigen::Map<Eigen::MatrixXd> dD(grad + cidx[n], k, n);
  dD.array() += lambda[0] * (inveps * D).array().tanh();

  // integration penalty gradient of D
  for (int i = 0; i < k; i++) {
    double a = sqrt(D.row(i).array().square().sum());
    if (a >= 1e-8) {
      dD.row(i).array() += lambda[1] * D.row(i).array() / a;
    }
  }

  Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, 1> T1(n);
  Eigen::Array<Eigen::MatrixXd, Eigen::Dynamic, 1> T2(n);
  for (int view = 0; view < n; view++) {
    Eigen::MatrixXd L1(k, p[view]);
    L1.setZero();
    // loss gradient of xi and loss gradient of D
    for (int block = 0; block < m; block++) {
      if (inds(block, 0) == view) {
        int other = inds(block, 1);
        Eigen::MatrixXd L2 =
            V(other) * (D.col(other).cwiseProduct(D.col(view))).asDiagonal();
        L1 += ((x[block] - V(view) * L2.transpose()).cwiseProduct(masks[block]) * L2)
                  .transpose();
        dD.col(view) +=
            (-2 * V(view).transpose() *
             (x[block] - V(view) * L2.transpose()).cwiseProduct(masks[block]) * V(other) *
             D.col(other).asDiagonal())
                .diagonal();
      } else if (inds(block, 1) == view) {
        int other = inds(block, 0);
        Eigen::MatrixXd L2(V(other) *
                           (D.col(other).cwiseProduct(D.col(view))).asDiagonal());
        L1 += ((x[block].transpose() - V(view) * L2.transpose())
                   .cwiseProduct(masks[block].transpose()) *
               L2)
                  .transpose();
        dD.col(view) += (-2 * V(view).transpose() *
                         (x[block].transpose() - V(view) * L2.transpose())
                             .cwiseProduct(masks[block].transpose()) *
                         V(other) * D.col(other).asDiagonal())
                            .diagonal();
      }
    }

    Eigen::MatrixXd VD(V(view) * D.col(view).asDiagonal());
    Eigen::MatrixXd VD2(
        (V(view) * (D.col(view).array().square().matrix()).asDiagonal()).transpose());
    if (lambda[3] > 0) {
      Eigen::Array<double, 1, Eigen::Dynamic> denoms(
          sqrt(VD.array().square().rowwise().sum()));
      for (int r = 0; r < p[view]; r++) {
        if (denoms(r) > 1e-8) {
          denoms(r) = 1 / denoms(r);
        } else {
          denoms(r) = 0.0;
        }
      }
      VD2.array().rowwise() *= denoms;
      dD.col(view) +=
          lambda[3] * ((V(view).array().colwise() * denoms.transpose()) * VD.array())
                          .matrix()
                          .colwise()
                          .sum()
                          .transpose();
    }
    T1(view) = Eigen::MatrixXd((inveps * VD).array().tanh().matrix());
    // its much cheaper to make the following sum and then get just one big
    // matrix multiplication per view instead of one per term in objective
    T2(view) =
        Eigen::MatrixXd(lambda[2] * (T1(view) * D.col(view).asDiagonal()).transpose() +
                        lambda[3] * VD2 - 2 * L1);
  }
  for (int view = 0; view < n; view++) {
    for (int i = 0; i < k; i++) {
      dD(i, view) += lambda[2] * (V(view).col(i).transpose() * T1(view).col(i))(0, 0);
    }
  }

  for (int c = 0; c < indices.cols(); c++) {
    int view = indices(0, c);
    int i = indices(1, c);
    int j = indices(2, c);
    Eigen::Map<const Eigen::MatrixXd> xi(theta + cidx[view], p[view], k);
    Eigen::Map<Eigen::MatrixXd> dxi(grad + cidx[view], p[view], k);
    double cx = cos(xi(i, j));
    double msx = -sin(xi(i, j));
    Eigen::VectorXd ti(T2(view).block(0, 0, k, i + 1) * pvss(view)(2)(i, j));
    Eigen::VectorXd tj(T2(view).block(0, 0, k, i + 1) * pvss(view)(3)(i, j));
    dxi(i, j) = (msx * (pvss(view)(0)(i, j) * ti) - cx * (pvss(view)(1)(i, j) * ti) +
                 cx * (pvss(view)(0)(i, j) * tj) + msx * (pvss(view)(1)(i, j) * tj))
                    .sum();
  }
}

void inv_v(double* xi, double* t, int n) {
  memset(xi, 0.0, n * n * sizeof(double));
  double* tnew = (double*)malloc(n * n * sizeof(double));
  double* f = (double*)malloc(n * n * sizeof(double));
  double* identity = (double*)calloc(n * n, sizeof(double));
  for (int i = 0; i < n; i++) {
    identity[i * n + i] = 1.0;
  }

  for (int v = n - 1; v >= 1; v--) {
    xi[v * n + (v - 1)] = atan2(t[(v - 1) * n + v], t[v * n + v]);
    if (v > 1) {
      for (int k = v - 2; k >= 0; k--) {
        double s = sin(xi[v * n + (k + 1)]);
        if (abs(t[(k + 1) * n + v] - s) < 1e-17) {
          if (v == n - 1 && abs(s) < 1e-17) {
            xi[v * n + k] = asin(t[k * n + v]);
          } else {
            xi[v * n + k] = 0.0;
          }
        } else {
          xi[v * n + k] = atan2(t[k * n + v], t[(k + 1) * n + v] / s);
        }
        if (xi[v * n + k] > M_PI / 2) xi[v * n + k] = xi[v * n + k] - M_PI;
        if (xi[v * n + k] < -M_PI / 2) xi[v * n + k] = xi[v * n + k] + M_PI;
      }
      memcpy(f, t, sizeof(double) * n * n);
      memcpy(tnew, identity, sizeof(double) * n * n);
      for (int k = v - 1; k >= 0; k--) {
        double c = cos(xi[v * n + k]);
        double s = sin(xi[v * n + k]);
        for (int j = 0; j < n; j++) {
          tnew[k * n + j] = c * t[k * n + j] - s * f[(k + 1) * n + j];
          f[k * n + j] = s * t[k * n + j] + c * f[(k + 1) * n + j];
        }
      }
      memcpy(t, tnew, sizeof(double) * n * n);
    }
    Rcpp::checkUserInterrupt();
  }

  free(identity);
  free(f);
  free(tnew);
}
