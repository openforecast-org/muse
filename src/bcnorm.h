// bcnorm.h — C++ analogue of greybox::dbcnorm (greybox/R/bcnorm.R:71-89).
//
// dbcnormLogDensity returns the log of the Box-Cox normal density at every
// element of q.  Same formula as the R version:
//   * lambda == 0:  density = dlnorm(q, meanlog = mu, sdlog = sigma)
//   * lambda == 1:  density = dnorm (q, mean = mu + 1, sd = sigma)
//   * otherwise:    density = q^(lambda-1) * 1/(sqrt(2 pi) sigma)
//                            * exp(-((q^lambda - 1)/lambda - mu)^2 / (2 sigma^2))
//                   density = 0 for q <= 0.
//
// The leading q^(lambda-1) factor (or implicitly 1/q for lambda = 0) IS the
// Box-Cox Jacobian, so the returned log-density is on the ORIGINAL response
// scale: summing over observations gives a log-likelihood comparable across
// lambdas.
//
// Not currently wired into estimation -- the engine still runs Gaussian MLE
// on the BC-transformed data.  This header is here as the C++ analogue of
// the R-side dbcnorm() call, so a future Python wrapper or alternative
// estimation path can compute the same BC-corrected log-likelihood from C++
// alone without re-implementing the formula.

#ifndef MUSE_BCNORM_H
#define MUSE_BCNORM_H

#include <cmath>
#include <armadillo>

// Per-element log-density.  q, mu must have the same length; sigma is a
// scalar.  Elements with q <= 0 are returned as -inf for lambda != 0,
// matching greybox's "density = 0" -> log(0) = -inf convention.
inline arma::vec bcnormLogDensity(const arma::vec& q,
                                  const arma::vec& mu,
                                  double sigma,
                                  double lambda){
    using namespace arma;
    const uword n = q.n_elem;
    vec out(n);
    const double LN_SQRT_2PI = 0.5 * std::log(2.0 * M_PI);

    if (std::abs(lambda) < 1e-12){
        // lambda == 0: dlnorm(q, meanlog = mu, sdlog = sigma)
        for (uword i = 0; i < n; ++i){
            if (q(i) <= 0.0 || !std::isfinite(q(i)) || !std::isfinite(mu(i))){
                out(i) = -datum::inf;
            } else {
                const double lq = std::log(q(i));
                const double z  = (lq - mu(i)) / sigma;
                out(i) = -lq - std::log(sigma) - LN_SQRT_2PI - 0.5 * z * z;
            }
        }
    } else if (std::abs(lambda - 1.0) < 1e-12){
        // lambda == 1: dnorm(q, mean = mu + 1, sd = sigma)
        for (uword i = 0; i < n; ++i){
            if (!std::isfinite(q(i)) || !std::isfinite(mu(i))){
                out(i) = -datum::inf;
            } else {
                const double z = (q(i) - (mu(i) + 1.0)) / sigma;
                out(i) = -std::log(sigma) - LN_SQRT_2PI - 0.5 * z * z;
            }
        }
    } else {
        // General lambda:
        //   log f(q) = (lambda-1)*log(q) - log(sigma) - 0.5*log(2 pi)
        //              - 0.5 * ((q^lambda - 1)/lambda - mu)^2 / sigma^2
        for (uword i = 0; i < n; ++i){
            if (q(i) <= 0.0 || !std::isfinite(q(i)) || !std::isfinite(mu(i))){
                out(i) = -datum::inf;
            } else {
                const double lq    = std::log(q(i));
                const double qlam  = std::pow(q(i), lambda);
                const double z     = ((qlam - 1.0) / lambda - mu(i)) / sigma;
                out(i) = (lambda - 1.0) * lq
                         - std::log(sigma) - LN_SQRT_2PI
                         - 0.5 * z * z;
            }
        }
    }
    return out;
}

#endif // MUSE_BCNORM_H
