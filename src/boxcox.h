/*************************
 Additional C++ functions not used in UComp
**************************/
// Global Box-Cox lambda search range.  Used everywhere a lambda value is
// clamped or filtered:
//   * preProcess (PTSmodel.h)         -- clamps user-supplied fixed lambda
//   * BSMclass::estim                  -- clamps post-BFGS lambdaStar
//   * snap-to-anchor selection         -- filters the anchor set
//   * BoxCoxEstim                      -- clamps Guerrero's estimate
// R can narrow the LOWER bound at runtime via SSinputs::lambdaLower
// (e.g. 1e-10 when y has zeros, 0 when y has negatives); the UPPER
// bound is currently fixed here.  Changing these constants takes effect
// everywhere at once -- there is no other source of [-1, 1] / [-2, 2]
// hardcoding.
constexpr double LAMBDA_BOUND_LOWER = -2.0;
constexpr double LAMBDA_BOUND_UPPER =  2.0;

// Struct for boxcox optimization
struct boxcoxInputs{
  vec y;
  int bunch;
};
/***************************************************
 * Function declarations
 ****************************************************/
// Box-Cox transfomration using Guerrero (1993)
vec BoxCox(vec, double);
// Inverse of Box-Cox transformation
vec invBoxCox(vec, double);
mat invBoxCoxMat(mat, double);
// Providing invBoxCox of vector and vector of variances
template <class T>
mat invBoxCox(vec&, T, double, double);
// Estimate lambda of Box-Cox transformation using Guerrero (1993)
double BoxCoxEstim(vec, int);
// Auxiliar function to estimate lambda of Box-Cox transform (Guerrero, 1993)
double auxBoxCox(vec&, void*);
// Gradient of auxiliar function to estimate lambda of Box-Cox tranformation
vec gradAuxBoxCox(vec&, void*, double, int&);
// Llik of residuals of fast signal decomposition
double llikDecompose(vec, vec, uvec&, string);
// Testing for no transformation, logs or box-cox transformation
double testBoxCox(vec, vec);
/***************************************************
 * Function implementations
 ****************************************************/
// Box-Cox transformation.  Branches are exact-equality on lambda
// (not threshold-based) so the LLIK / AIC is continuous in lambda and
// the only shortcuts are the two singular points where the general
// formula is undefined:
//   lambda == 1  -> identity   (general formula gives (y - 1)/1 = y - 1,
//                               which differs from y by a constant; we
//                               drop that constant so the Kalman fit
//                               matches greybox's `dbcnorm(lambda=1)`
//                               convention -- see ARCHITECTURE.md)
//   lambda == 0  -> log(y)     (general formula has 0/0)
vec BoxCox(vec y, double lambda){
  if (lambda == 1.0)
      return y;
  if (any(y < 0)){
    throw std::invalid_argument( "BoxCox transformation impossible: negative values encountered!!" );
  }
  if (lambda == 0.0){
      return log(y);
  } else {
    // return (pow(sign(y) % abs(y), lambda) - 1) / lambda;
      return (pow(y, lambda) - 1) / lambda;
  }
}
// Inverse of Box-Cox transformation
vec invBoxCox(vec y, double lambda){
  if (lambda == 0.0){
      return exp(y);
  } else if (lambda == 1.0) {
      return y;
  } else {
    //vec aux = y * lambda + 1;
    //return sign(aux) % pow(abs(aux), 1 / lambda);
      return pow(y * lambda + 1, 1 / lambda);
  }
}
mat invBoxCoxMat(mat y, double lambda){
  if (lambda == 0.0){
      return exp(y);
  } else if (lambda == 1.0) {
      return y;
  } else {
    //vec aux = y * lambda + 1;
    //return sign(aux) % pow(abs(aux), 1 / lambda);
      return pow(y * lambda + 1, 1 / lambda);
  }
}
// Estimate lambda of Box-Cox transformation using Guerrero (1993)
double BoxCoxEstim(vec y, int bunch){
  // quasiNewton(std::function <double (vec& x, void* inputs)> objFun,
  //             std::function <vec (vec& x, void* inputs, double obj, int& nFuns)> gradFun,
  //             vec& xNew, void* inputs, double& objNew, vec& gradNew, mat& iHess);
  vec lambda(1), grad(1);
  double obj;
  mat ihess(1, 1);
  boxcoxInputs inputs;
  inputs.y = y;
  inputs.bunch = bunch;
  lambda(0) = 0.0;
  quasiNewton(auxBoxCox, gradAuxBoxCox, lambda, &inputs, obj, grad, ihess, false);
  if (lambda(0) < LAMBDA_BOUND_LOWER){
    lambda(0) = LAMBDA_BOUND_LOWER;
  } else if (lambda(0) > LAMBDA_BOUND_UPPER){
    lambda(0) = LAMBDA_BOUND_UPPER;
  }
  return lambda(0);
}
// Auxiliar function to estimate lambda of Box-Cox transform (Guerrero, 1993)
double auxBoxCox(vec& lambda, void* inputs_data){
  boxcoxInputs* inputs = (boxcoxInputs*)inputs_data;
  int n = inputs->y.n_elem;
  int ny = n / inputs->bunch;
  mat yr = conv_to<mat>::from(inputs->y);
  yr.set_size(inputs->bunch, ny);
  rowvec myr = nanMean(yr);
  rowvec sd = nanStddev(yr);
  rowvec yt = sd / pow(myr, 1 - lambda(0));
  return stddev(yt) / mean(yt);
}
// Auxiliar gradient function to estimate lambda of Box-Cox transform
vec gradAuxBoxCox(vec& p, void* inputs_data, double obj0, int& nFuns){
  double obj;
  int nPar = p.n_elem;
  vec grad(nPar), p0 = p, inc;
  nFuns = 0;
  inc = 1e-8;
  for (int i = 0; i < nPar; i++){
    p0 = p;
    p0.row(i) += inc;
    obj = auxBoxCox(p0, inputs_data);
    grad.row(i) = (obj - obj0) / inc;
  }
  nFuns += nPar;
  return grad;
}
// Llik of residuals of fast signal decomposition
double llikDecompose(vec y, vec periods, uvec& ind, string type){
    // ind is vector of indices of output variable used in likelihood computation
    // type: type of model 'hr' harmonic regression on full data,
    //       otherwise it does MA trend and harmonic regression on the detrended
    vec sigma2, beta, stdBeta, e, dy = y, yFit;
    uword ma2 = 0;
    double seas = max(periods);
    mat u(0, 0);
    if (type == "hr"){
        harmonicRegress(y, u, periods, 3, beta, stdBeta, e, yFit);
        ind = find_finite(y);
        e = e(ind);
        sigma2 = (e.t() * e) / e.n_elem;
        return -((double)e.n_elem / 2) * (log(2 * datum::pi * sigma2(0)) + 1);
    }
    if (y.n_elem > 14 && y.is_finite()){
        // Detrending with MA
        double ma = seas;
        if (seas / 2 == floor(seas / 2))
            ma = seas + 1;
        if (ma < 5)
            ma = 5;
        ma2 = (ma - 1) / 2;
        vec MA(ma, fill::value(1 / ma));
        e = conv(MA, y);
        y = y.rows(ma2, y.n_rows - ma2 - 1);
        e = e.rows(ma - 1, e.n_rows - ma);
        dy = y - e;
    }
    harmonicRegress(dy, u, periods, 0, beta, stdBeta, e, yFit);
    ind = find_finite(e);
    e = e(ind);
    ind += ma2;
    sigma2 = (e.t() * e) / e.n_elem;
    return -((double)e.n_elem / 2) * (log(2 * datum::pi * sigma2(0)) + 1);
}
// Testing for no transformation, logs or box-cox transformation.
//
// Returns the Box-Cox λ seed for joint-BFGS estimation.  Compares the
// log-likelihood of a simple decomposition fit under three candidates
// (λ = 1 / λ = 0 / λ = BoxCoxEstim) and returns the winner.
//
// NaN sentinel convention: every "decomposition failed" branch maps to
// a large *negative* value so the failed candidate loses every
// subsequent `bestLLIK < cLLIK` comparison.  An earlier version of this
// function used positive sentinels (1e8, 1e9) for the log / BoxCox-aux
// candidates, which caused failed candidates to spuriously win and
// pinned λ to 0 whenever the input contained NaN entries.
double testBoxCox(vec y, vec periods){
    // https://stats.stackexchange.com/questions/261380/how-do-i-get-the-box-cox-log-likelihood-using-the-jacobian
    string typeDecompose = "ma";   // hr: harmonic regression; or "ma": moving average
    double lambda = 1.0;
    // Repair NaN / Inf entries in y up front so `llikDecompose` and
    // BoxCoxEstim work on a finite series.  Naive filtering would break
    // seasonal periodicity (BoxCoxEstim's variance-stabilisation test
    // assumes regularly-spaced data), so we forward-fill missing
    // entries with the most recent finite neighbour, then back-fill any
    // leading NaN with the first finite value.  The decomposition test
    // is a coarse warm-start estimator; nearest-neighbour fill is plenty.
    if (!y.is_finite()){
        uword n = y.n_elem;
        double last = arma::datum::nan;
        for (uword i = 0; i < n; ++i){
            if (std::isfinite(y(i))) last = y(i);
            else if (std::isfinite(last)) y(i) = last;
        }
        // back-fill any leading NaN
        last = arma::datum::nan;
        for (int i = (int)n - 1; i >= 0; --i){
            if (std::isfinite(y(i))) last = y(i);
            else if (std::isfinite(last)) y(i) = last;
        }
        if (!y.is_finite()) return lambda;   // truly all-NaN
    }
    if (y.n_elem < 4) return lambda;
    if (any(y) < 1){
        return lambda;
    }
    // When transformation is possible
    uvec ind;
    // No transformation (λ = 1)
    double bestLLIK = llikDecompose(y, periods, ind, typeDecompose);
    if (isnan(bestLLIK))
        bestLLIK = -1e10;
    vec cLLIK(1);
    // Log transform (λ = 0)
    double aux = BoxCoxEstim(y, std::max(2.0, max(periods)));
    cLLIK(0) = llikDecompose(log(y), periods, ind, typeDecompose) - sum(log(y(ind)));
    if (isnan(cLLIK(0)))
        cLLIK(0) = -1e20;       // failed: must lose the next comparison
    if (bestLLIK < cLLIK(0)){
        lambda = 0.0;
        bestLLIK = cLLIK(0);
    }
    // Box-Cox transform at the BoxCoxEstim-recommended λ = aux
    if (abs(aux) > 0.1 && aux < 0.9){
        cLLIK(0) = llikDecompose(BoxCox(y, aux), periods, ind, typeDecompose) + sum(log(pow(y(ind), aux - 1)));
        if (isnan(cLLIK(0)))
            cLLIK(0) = -1e20;   // failed: must lose
        if (bestLLIK < cLLIK(0))
            lambda = aux;
    }
    return lambda;
}
