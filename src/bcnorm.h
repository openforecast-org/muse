// bcnorm.h — C++ analogue of greybox::dbcnorm (greybox/R/bcnorm.R:71-89),
// aligned with the engine's BoxCox thresholds (src/boxcox.h:35-49) so the
// caller doesn't need to special-case lambda == 1.
//
// bcnormLogDensity returns the log-density of the Box-Cox-normal distribution
// at every element of q, using the same three branches the engine itself
// applies for BoxCox:
//   * |lambda| < 0.02  (engine treats as log)
//        log f(q) = -log(q) - log(sigma) - 0.5 log(2 pi)
//                   - 0.5 * ((log q - mu) / sigma)^2
//       (= dlnorm(q, meanlog = mu, sdlog = sigma, log = TRUE))
//   * lambda > 0.98    (engine treats as identity, NOT formal Box-Cox lambda=1)
//        log f(q) = -log(sigma) - 0.5 log(2 pi)
//                   - 0.5 * ((q - mu) / sigma)^2
//       (= dnorm(q, mean = mu, sd = sigma, log = TRUE))
//   * otherwise (Box-Cox proper)
//        log f(q) = (lambda - 1) * log(q) - log(sigma) - 0.5 log(2 pi)
//                   - 0.5 * (((q^lambda - 1)/lambda - mu) / sigma)^2
//
// The first term (lambda - 1) * log(q) is the Box-Cox Jacobian; summing this
// log-density over the in-sample observations gives the joint log-likelihood
// on the ORIGINAL response scale (comparable across lambdas and to
// dbcnorm-based likelihoods elsewhere in the ecosystem).
//
// Used by the BSMclass estimation path in src/PTSmodel.h to compute the
// reported LLIK on the original response scale, replacing the previous
// hand-rolled Gaussian formula that lived on the BC scale.

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

    if (std::abs(lambda) < 0.02){
        // Engine's log/lognormal branch (matches src/boxcox.h:40).
        // dlnorm(q, meanlog = mu, sdlog = sigma)
        for (uword i = 0; i < n; ++i){
            if (q(i) <= 0.0 || !std::isfinite(q(i)) || !std::isfinite(mu(i))){
                out(i) = -datum::inf;
            } else {
                const double lq = std::log(q(i));
                const double z  = (lq - mu(i)) / sigma;
                out(i) = -lq - std::log(sigma) - LN_SQRT_2PI - 0.5 * z * z;
            }
        }
    } else if (lambda > 0.98){
        // Engine's identity branch (matches src/boxcox.h:35).
        // dnorm(q, mean = mu, sd = sigma) -- NO mu + 1 shift, since the
        // engine never transformed y in this band.
        for (uword i = 0; i < n; ++i){
            if (!std::isfinite(q(i)) || !std::isfinite(mu(i))){
                out(i) = -datum::inf;
            } else {
                const double z = (q(i) - mu(i)) / sigma;
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
