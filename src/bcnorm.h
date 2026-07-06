// bcnorm.h — C++ implementation of the Box-Cox normal distribution,
// matching greybox::dbcnorm (greybox/R/bcnorm.R).
//
// The BCnorm density for an observation y with location mu, scale sigma,
// and Box-Cox parameter lambda is
//
//   f(y; mu, sigma, lambda)
//       = |g'(y; lambda)| * phi((g(y; lambda) - mu) / sigma) / sigma
//
// where g(y; lambda) = BoxCox(y, lambda) is the Box-Cox transform and
// phi is the standard normal density.
//
// Three branches, consistent with src/boxcox.h.  Exact-equality
// switches (not thresholds): the general formula is well-defined for
// every lambda except the two singular points, and using thresholds
// introduces an artificial AIC discontinuity near them.
//   lambda == 1    (identity)  g(y) = y,       g'(y) = 1     => log-Jacobian = 0
//   lambda == 0    (log)       g(y) = log(y),  g'(y) = 1/y   => log-Jacobian = -log(y)
//   otherwise      (general)   g(y) = (y^lambda-1)/lambda,
//                              g'(y) = y^(lambda-1)          => log-Jacobian = (lambda-1)*log(y)
//
// Public API
// ----------
// bcnormBoxCox(y, lambda)           scalar Box-Cox (mirrors src/boxcox.h)
// bcnormLogJac(y, lambda)           log |dg/dy|: used to accumulate the
//                                    per-observation BCnorm Jacobian in llik()
// bcnormLogDensityScalar(y, mu,     log BCnorm density for one observation;
//                        sigma,      mu is the KF-predicted BoxCox(y, lambda)
//                        lambda)     value (= Z * a_t), sigma = sqrt(innVar*F)
// bcnormLogDensity(q, mu, sigma,    vectorised version (kept for external use)
//                  lambda)

#ifndef MUSE_BCNORM_H
#define MUSE_BCNORM_H

#include <cmath>
#include <limits>
#include <armadillo>

// -----------------------------------------------------------------------
// Scalar Box-Cox transform — same three-branch logic as vec BoxCox() in
// src/boxcox.h so the density is consistent with the KF data transform.
// -----------------------------------------------------------------------
inline double bcnormBoxCox(double y, double lambda){
    if (lambda == 1.0)    return y;
    if (lambda == 0.0)    return std::log(y);
    return (std::pow(y, lambda) - 1.0) / lambda;
}

// -----------------------------------------------------------------------
// Log Jacobian of the Box-Cox transform at y:  log |d(g(y))/dy|
// Returns 0 for non-positive or non-finite y (density = 0 there).
// -----------------------------------------------------------------------
inline double bcnormLogJac(double y, double lambda){
    if (y <= 0.0 || !std::isfinite(y)) return 0.0;
    if (lambda == 1.0)               return 0.0;              // identity: g'=1
    if (lambda == 0.0)               return -std::log(y);     // log: g'=1/y
    return (lambda - 1.0) * std::log(y);                      // general: g'=y^(lambda-1)
}

// -----------------------------------------------------------------------
// Vectorised log Jacobian: element-wise bcnormLogJac over a vector of y.
// -----------------------------------------------------------------------
inline arma::vec bcnormLogJac(const arma::vec& y, double lambda){
    arma::vec out(y.n_elem);
    for (arma::uword i = 0; i < y.n_elem; ++i)
        out(i) = bcnormLogJac(y(i), lambda);
    return out;
}

// -----------------------------------------------------------------------
// Log BCnorm density for one observation.
//
//   y_raw   original (pre-BoxCox) observation
//   mu      KF-predicted value of g(y_raw) = BoxCox(y_raw, lambda), i.e. Z*a_t
//   sigma   innovation std dev = sqrt(innVariance * F_t)
//   lambda  Box-Cox parameter
//
// Returns log f(y_raw; mu, sigma, lambda)
//       = bcnormLogJac(y_raw, lambda) + log phi((g(y_raw)-mu)/sigma) - log sigma
// -----------------------------------------------------------------------
inline double bcnormLogDensityScalar(double y_raw, double mu,
                                      double sigma, double lambda){
    // arma::datum::pi rather than M_PI: the latter is not defined by MSVC's
    // <cmath> unless _USE_MATH_DEFINES is set before the include.
    static const double LN_SQRT_2PI = 0.5 * std::log(2.0 * arma::datum::pi);

    if (!std::isfinite(y_raw) || !std::isfinite(mu) || sigma <= 0.0)
        return -std::numeric_limits<double>::infinity();
    // Box-Cox domain.  g(y) = (y^lambda - 1)/lambda is finite for:
    //   * y > 0          (all lambda),
    //   * y == 0 with lambda > 0   (g(0) = -1/lambda, e.g. sqrt(0) = 0),
    //   * any real y at lambda == 1 (identity).
    // It is UNDEFINED only for y <= 0 with lambda <= 0 (log(0) at lambda 0,
    // 0^negative -> Inf at lambda < 0).  y < 0 with fractional lambda gives a
    // complex g, caught by the !isfinite(g) check below.  So reject only the
    // genuinely-undefined corner here -- in particular do NOT reject y == 0 for
    // lambda > 0, which is exactly the variance-stabilising case for
    // intermittent / zero-containing series (sqrt etc.).
    if (lambda <= 0.0 && y_raw <= 0.0)
        return -std::numeric_limits<double>::infinity();

    const double g = bcnormBoxCox(y_raw, lambda);
    if (!std::isfinite(g))
        return -std::numeric_limits<double>::infinity();

    const double z = (g - mu) / sigma;
    return bcnormLogJac(y_raw, lambda)
           - std::log(sigma) - LN_SQRT_2PI - 0.5 * z * z;
}

// -----------------------------------------------------------------------
// Vectorised log BCnorm density.  q, mu and sigma are element-wise: each
// observation has its OWN predictive standard deviation sigma(i) (the
// state-space innovation sd sqrt(innVar * F_t) is heteroscedastic).  This is
// the form the likelihood needs -- call it once over all observations and sum.
// -----------------------------------------------------------------------
inline arma::vec bcnormLogDensity(const arma::vec& q,
                                  const arma::vec& mu,
                                  const arma::vec& sigma,
                                  double lambda){
    const arma::uword n = q.n_elem;
    arma::vec out(n);
    for (arma::uword i = 0; i < n; ++i)
        out(i) = bcnormLogDensityScalar(q(i), mu(i), sigma(i), lambda);
    return out;
}

#endif // MUSE_BCNORM_H
