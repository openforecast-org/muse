// musecpp2R.cpp - thin R binding for the muse engine.
//
// This file is the *only* place in the package that knows about Rcpp.  It
// turns SEXP arguments into a MuseInputs struct, calls runMuseCommand(),
// and packs the resulting MuseOutputs struct into an Rcpp::List.  All the
// dispatch logic lives in musecore.h and is Rcpp-free, so a Python binding
// (pybind11 / nanobind) can reuse runMuseCommand() unchanged.

// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;
using namespace Rcpp;
using std::string;
#include "musecore.h"

// Internal-only Rcpp entry point.  The dot-prefixed name keeps the R
// wrapper out of `ls("package:muse")`, autocomplete, and the package-
// level `muse::` lookup; package code calls it via `.UCompC(...)`.
// Users should never call this directly — `pts()` is the only
// supported R-side entry point.
// [[Rcpp::export(.UCompC)]]
SEXP UCompC(SEXP commands, SEXP ys, SEXP us, SEXP models, SEXP hs,
            SEXP lambdas, SEXP outliers, SEXP tTests, SEXP criterions,
            SEXP periodss, SEXP rhoss, SEXP verboses, SEXP stepwises,
            SEXP p0s, SEXP armas, SEXP TVPs, SEXP seass,
            SEXP trendOptionss, SEXP seasonalOptionss,
            SEXP irregularOptionss,
            SEXP nsims, SEXP seeds){

    // --- Marshall SEXP -> MuseInputs (no engine logic in this file) ---
    MuseInputs in;
    NumericVector yr(ys), periodsr(periodss), rhosr(rhoss), p0r(p0s), TVPr(TVPs);
    NumericMatrix ur(us);

    in.command          = CHAR(STRING_ELT(commands, 0));
    in.y                = vec(yr.begin(),       yr.size(),      false);
    in.u                = mat(ur.begin(),       ur.nrow(), ur.ncol(), false);
    in.model            = CHAR(STRING_ELT(models, 0));
    in.h                = as<int>(hs);
    in.lambda           = as<double>(lambdas);
    in.outlier          = as<double>(outliers);
    in.tTest            = as<bool>(tTests);
    in.criterion        = CHAR(STRING_ELT(criterions, 0));
    in.periods          = vec(periodsr.begin(), periodsr.size(), false);
    in.rhos             = vec(rhosr.begin(),    rhosr.size(),    false);
    in.verbose          = as<bool>(verboses);
    in.stepwise         = as<bool>(stepwises);
    in.p0               = vec(p0r.begin(),      p0r.size(),      false);
    in.armaFlag         = as<bool>(armas);
    in.TVP              = vec(TVPr.begin(),     TVPr.size(),     false);
    in.seas             = as<double>(seass);
    in.trendOptions     = CHAR(STRING_ELT(trendOptionss, 0));
    in.seasonalOptions  = CHAR(STRING_ELT(seasonalOptionss, 0));
    in.irregularOptions = CHAR(STRING_ELT(irregularOptionss, 0));
    in.nsim             = as<int>(nsims);
    in.seed             = static_cast<unsigned>(as<int>(seeds));

    // --- Run the engine ---
    MuseOutputs out;
    runMuseCommand(in, out);

    if (out.isError)
        return List::create(Named("model") = "error");

    // --- Pack MuseOutputs -> Rcpp::List ---
    List output = List::create(
        Named("model")    = out.model,
        Named("yFor")     = out.yFor,
        Named("h")        = out.h,
        Named("yForV")    = out.yForV,
        Named("estimOk")  = out.estimOk,
        Named("lambda")          = out.lambda,
        Named("lambdaEstimated") = out.lambdaEstimated,
        Named("objFunValue")     = out.objFunValue,
        Named("periods")  = out.periods,
        Named("rhos")     = out.rhos,
        Named("p")        = out.p,
        Named("p0")       = out.p0Return,
        Named("parNames") = out.parNames,
        Named("ns")       = out.ns,
        Named("criteria") = out.criteria);

    if (out.hasValidate){
        output("table")        = out.table;
        output("v")            = out.v;
        output("covp")         = out.covp;
        output("coef")         = out.coef;
        output("typeOutliers") = out.typeOutliers;
    }
    if (out.hasFilter){
        output("a")          = out.a;
        output("P")          = out.P;
        output("v")          = out.v;
        output("ns")         = out.ns;
        output("yFitV")      = out.yFitV;
        output("yFit")       = out.yFit;
        output("eps")        = out.eps;
        output("eta")        = out.eta;
        output("stateNames") = out.stateNamesStr;
    }
    if (out.hasComponents){
        output("comp")      = out.comp;
        output("compV")     = out.compV;
        output("m")         = out.m;
        output("compNames") = out.compNames;
    }
    if (out.hasSimulate){
        output("simPaths") = out.simPaths;   // h x nsim, original-scale
    }
    return output;
}

// Standalone ARMA / SARMA fitter — used by R/pts-translate.R::
// .pts_select_arma() in place of stats::arima.  Fits any per-lag ARMA
// (non-seasonal arma(p, q) when length(arOrders) == 1, SARMA(p, q)(P, Q)_s
// when length == 2) to the supplied vector via the same SSmodel /
// quasi-Newton machinery PTS uses, with the same MLE σ̂² convention and
// BFGS objective shape.  ACF/PACF asymmetric init is applied so the
// optimiser doesn't drift to the cancellation manifold.
//
// ARMA(0, 0) short-circuits to var(y) with the same IC formula
// (k = 1 + 0 + 0 = 1) as a fitted ARMA(p, q) — so the grid search can
// rank the no-ARMA cell against any ARMA cell on equal terms.
//
// Inputs:
//   ys        — numeric vector (BC-scale residuals from PTS).
//   arOrders  — integer vector, AR orders per lag block.
//   maOrders  — integer vector, MA orders per lag block.
//   armaLags  — integer vector of lags (always c(1) or c(1, s)).
//   criterion — accepted for API symmetry, unused (all four ICs are
//                returned).
// Output: List with logLik / AIC / AICc / BIC / BICc / coef (AR blocks
// concatenated, then MA blocks, both natural scale) / sigma2 / succeed.
//
// Internal-only Rcpp entry point.  Dot-prefixed for the same reason
// as `.UCompC` — used by R/pts-translate.R::.pts_select_arma() as the
// per-cell ARMA fitter during the IC grid search.  Not part of the
// user-facing API.
// [[Rcpp::export(.UCompARMAC)]]
SEXP UCompARMAC(SEXP ys, SEXP arOrders_, SEXP maOrders_, SEXP armaLags_,
                SEXP criterion_){
    (void)criterion_;     // unused — wrapper returns all four ICs, R picks
    NumericVector yr(ys);
    vec y(yr.begin(), yr.size(), false);
    IntegerVector arOrdersR(arOrders_);
    IntegerVector maOrdersR(maOrders_);
    IntegerVector armaLagsR(armaLags_);
    ivec arOrders(arOrdersR.size()), maOrders(maOrdersR.size()),
         armaLags(armaLagsR.size());
    for (R_xlen_t i = 0; i < arOrdersR.size(); ++i) arOrders(i) = arOrdersR[i];
    for (R_xlen_t i = 0; i < maOrdersR.size(); ++i) maOrders(i) = maOrdersR[i];
    for (R_xlen_t i = 0; i < armaLagsR.size(); ++i) armaLags(i) = armaLagsR[i];

    int arFree = (arOrders.n_elem > 0) ? (int)arma::sum(arOrders) : 0;
    int maFree = (maOrders.n_elem > 0) ? (int)arma::sum(maOrders) : 0;
    if (arFree < 0) arFree = 0;
    if (maFree < 0) maFree = 0;

    // Drop non-finite entries up front so the KF doesn't crash on NaN.
    uvec ok = find_finite(y);
    vec yClean = y(ok);
    if (yClean.n_elem < (uword)std::max(4, arFree + maFree + 2))
        return List::create(Named("succeed") = false);

    int n_finite = (int)yClean.n_elem;

    // SSinputs setup mirroring BSMclass's pattern at musecore.h:170+.
    SSinputs ssIn;
    ssIn.y          = yClean;
    ssIn.y_raw      = yClean;
    ssIn.lambda     = 1.0;          // residuals already on the BC scale
    ssIn.u          = mat(0, yClean.n_elem);
    ssIn.cLlik      = true;         // concentrated MLE σ̂²
    ssIn.augmented  = false;        // ARMA is stationary, no diffuse init
    ssIn.exact      = (arFree == 0);
    ssIn.h          = 0;
    ssIn.verbose    = false;
    ssIn.userInputs = nullptr;
    ssIn.estimateLambda = false;

    // ARMA(0, 0) — fit σ̂² in closed form and compute the IC with the
    // SAME formula the engine uses for any ARMA(p, q): k = p.n_elem +
    // nonStationaryTerms = 1.  Keeping the comparison IC-fair across the
    // grid so the (0, 0) cell isn't artificially favoured / penalised.
    if (arFree == 0 && maFree == 0){
        double sigma2 = arma::var(yClean, 1);     // 1/n divisor (MLE)
        double LL = -0.5 * n_finite *
                    (std::log(2.0 * datum::pi) + std::log(sigma2) + 1.0);
        int k = 1;     // σ² only
        double aic  = -2.0 * LL + 2.0 * k;
        double bic  = -2.0 * LL + std::log((double)n_finite) * k;
        double aicc = aic + (2.0 * k * (k + 1)) /
                            std::max(1, n_finite - k - 1);
        double bicc = bic + (std::log((double)n_finite) * k * (k + 1)) /
                            std::max(1, n_finite - k - 1);
        // For ARMA(0, 0) the "residuals" are the input series minus its
        // mean — no model, no innovations.  Return the centred series so
        // the cascade can pass through cleanly.
        double mu = arma::mean(yClean);
        NumericVector resids(yClean.n_elem);
        for (uword t = 0; t < yClean.n_elem; ++t) resids[t] = yClean(t) - mu;
        return List::create(
            Named("logLik")    = LL,
            Named("AIC")       = aic,
            Named("AICc")      = aicc,
            Named("BIC")       = bic,
            Named("BICc")      = bicc,
            Named("coef")      = NumericVector(),
            Named("sigma2")    = sigma2,
            Named("residuals") = resids,
            Named("succeed")   = true);
    }

    // ACF/PACF asymmetric init — per-lag layout matching armaMatrices()'s
    // consumption order:
    //   p[0]                       = log-stddev seed
    //   p[1..arFree]               = AR coefs, block-by-block
    //   p[arFree+1..arFree+maFree] = MA coefs, block-by-block
    int maxLag = 0;
    for (uword b = 0; b < arOrders.n_elem; ++b)
        maxLag = std::max(maxLag, arOrders(b) * armaLags(b));
    for (uword b = 0; b < maOrders.n_elem; ++b)
        maxLag = std::max(maxLag, maOrders(b) * armaLags(b));
    vec pacf = (maxLag > 0) ? sampleYWpacf(yClean, maxLag) : vec();
    vec acf  = (maxLag > 0) ? sampleACF (yClean, maxLag) : vec();
    auto clamp = [](double v, double fb){
        if (!std::isfinite(v)) return fb;
        if (v >  0.85) return  0.85;
        if (v < -0.85) return -0.85;
        return v;
    };

    vec p0(arFree + maFree + 1, fill::zeros);
    p0(0) = -1.0;
    uword pos = 1;
    for (uword b = 0; b < arOrders.n_elem; ++b){
        int pi = arOrders(b);
        int Li = armaLags(b);
        for (int j = 0; j < pi; ++j){
            int lagIdx = (j + 1) * Li;
            double seed = (lagIdx <= (int)pacf.n_elem) ? pacf(lagIdx - 1) : 0.1;
            p0(pos++) = clamp(seed, 0.1);
        }
    }
    uword maStart = pos;
    for (uword b = 0; b < maOrders.n_elem; ++b){
        int qi = maOrders(b);
        int Li = armaLags(b);
        for (int j = 0; j < qi; ++j){
            int lagIdx = (j + 1) * Li;
            double seed = (lagIdx <= (int)acf.n_elem) ? acf(lagIdx - 1) : -0.1;
            p0(pos++) = clamp(seed, -0.1);
        }
    }
    // Tie-breaker on the leading (AR_1, MA_1) pair when arOrders[0] > 0 and
    // maOrders[0] > 0 — pushes MA off the φ_i == θ_i manifold.
    if (arOrders.n_elem > 0 && maOrders.n_elem > 0 &&
        arOrders(0) > 0 && maOrders(0) > 0){
        double a = p0(1);
        double m = p0(maStart);
        if (std::abs(a - m) < 0.1){
            double offset = (m > 0.0) ? -0.2 : 0.2;
            p0(maStart) = clamp(m + offset, -0.1);
        }
    }

    // Construct ARMA model with per-lag SARMA support.
    ARMAmodel sys(ssIn, arOrders, maOrders, armaLags);

    SSinputs sysIn = sys.SSmodel::getInputs();
    sysIn.llikFUN = llik;
    sysIn.p0      = p0;
    sys.SSmodel::setInputs(sysIn);

    sys.SSmodel::estim();

    SSinputs sysOut = sys.SSmodel::getInputs();
    vec criteria = sysOut.criteria;   // {LLIK, AIC, BIC, AICc, BICc}
    if (criteria.n_elem < 5)
        return List::create(Named("succeed") = false);
    double LL   = criteria(0);
    double aic  = criteria(1);
    double bic  = criteria(2);
    double aicc = criteria(3);
    double bicc = criteria(4);

    // Convert the BFGS p-vector into natural-scale AR / MA coefficients
    // per block.  polyStationary is applied per-block (matching how
    // armaMatrices() reads its slice).
    NumericVector coef(arFree + maFree);
    {
        uword pIn  = 1;
        uword pOut = 0;
        for (uword b = 0; b < arOrders.n_elem; ++b){
            int pi = arOrders(b);
            if (pi == 0) continue;
            vec block = sysOut.p.rows(pIn, pIn + pi - 1);
            polyStationary(block);
            for (int j = 0; j < pi; ++j) coef[pOut++] = -block(j);
            pIn += pi;
        }
        for (uword b = 0; b < maOrders.n_elem; ++b){
            int qi = maOrders(b);
            if (qi == 0) continue;
            vec block = sysOut.p.rows(pIn, pIn + qi - 1);
            polyStationary(block);
            for (int j = 0; j < qi; ++j) coef[pOut++] = block(j);
            pIn += qi;
        }
        (void)pOut;
    }
    double sigma2 = sysOut.innVariance;

    // Innovations (= one-step-ahead residuals).  Pack as a NumericVector
    // so the R-side cascade can feed them back into the next-lag ARMA fit
    // without round-tripping through a refit.
    NumericVector resids(sysOut.v.n_elem);
    for (uword t = 0; t < sysOut.v.n_elem; ++t) resids[t] = sysOut.v(t);

    return List::create(
        Named("logLik")    = LL,
        Named("AIC")       = aic,
        Named("AICc")      = aicc,
        Named("BIC")       = bic,
        Named("BICc")      = bicc,
        Named("coef")      = coef,
        Named("sigma2")    = sigma2,
        Named("residuals") = resids,
        Named("succeed")   = std::isfinite(LL));
}
