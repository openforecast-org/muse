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

// [[Rcpp::export]]
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
        output("table") = out.table;
        output("v")     = out.v;
        output("covp")  = out.covp;
        output("coef")  = out.coef;
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

// Standalone ARMA fitter — used by R/pts-translate.R::.pts_select_arma()
// in place of stats::arima.  Fits a (non-seasonal) ARMA(ar, ma) to the
// supplied vector via the same SSmodel / quasi-Newton machinery PTS uses,
// with the same MLE σ̂² convention and BFGS objective shape.  ACF/PACF
// asymmetric init is applied so the optimiser doesn't drift to the
// cancellation manifold (the same fix used in PTSmodel.h:3662).
//
// Inputs:
//   ys        — numeric vector (the BC-scale residuals from PTS).
//   ar, ma    — non-negative scalars.
//   criterion — "aic" / "aicc" / "bic" / "bicc" (unused inside; the wrapper
//                returns all four ICs and R picks the relevant one).
// Output: List with logLik / AIC / AICc / BIC / BICc / coef (AR coefs first,
// then MA coefs, both natural scale) / sigma2 / succeed.
//
// [[Rcpp::export]]
SEXP UCompARMAC(SEXP ys, SEXP ar_, SEXP ma_, SEXP criterion_){
    (void)criterion_;     // unused — wrapper returns all four ICs, R picks
    NumericVector yr(ys);
    vec y(yr.begin(), yr.size(), false);
    int ar = as<int>(ar_);
    int ma = as<int>(ma_);
    if (ar < 0) ar = 0;
    if (ma < 0) ma = 0;

    // Drop non-finite entries up front so the KF doesn't crash on NaN.
    uvec ok = find_finite(y);
    vec yClean = y(ok);
    if (yClean.n_elem < std::max(4, ar + ma + 2))
        return List::create(Named("succeed") = false);

    // SSinputs setup mirroring BSMclass's pattern at musecore.h:170+.
    SSinputs ssIn;
    ssIn.y          = yClean;
    ssIn.y_raw      = yClean;
    ssIn.lambda     = 1.0;          // residuals already on the BC scale
    ssIn.u          = mat(0, yClean.n_elem);
    ssIn.cLlik      = true;         // concentrated MLE σ̂²
    ssIn.augmented  = false;        // ARMA is stationary, no diffuse init
    ssIn.exact      = (ar == 0);
    ssIn.h          = 0;
    ssIn.verbose    = false;
    ssIn.userInputs = nullptr;
    ssIn.estimateLambda = false;

    if (ar == 0 && ma == 0){
        // Pure-noise fit — σ̂² is just var(y), no optimisation needed.
        double sigma2 = arma::var(yClean, 1);    // 1/n divisor (MLE)
        int n = (int)yClean.n_elem;
        double LL = -0.5 * n * (std::log(2.0 * datum::pi) +
                                std::log(sigma2) + 1.0);
        int k = 1;       // only σ²
        double aic  = -2.0 * LL + 2.0 * k;
        double bic  = -2.0 * LL + std::log((double)n) * k;
        double aicc = aic + (2.0 * k * (k + 1)) / std::max(1, n - k - 1);
        double bicc = bic + (std::log((double)n) * k * (k + 1)) /
                            std::max(1, n - k - 1);
        return List::create(
            Named("logLik")  = LL,
            Named("AIC")     = aic,
            Named("AICc")    = aicc,
            Named("BIC")     = bic,
            Named("BICc")    = bicc,
            Named("coef")    = NumericVector(),
            Named("sigma2")  = sigma2,
            Named("succeed") = true);
    }

    // ACF/PACF asymmetric init — same logic as PTSmodel.h:3662, simplified
    // for the scalar (ar, ma) case.  BFGS-scale slot layout:
    //   p[0]            = log-stddev seed
    //   p[1..ar]        = AR coefs in the polyStationary input space (PACF)
    //   p[ar+1..ar+ma]  = MA coefs in the polyStationary input space
    int maxLag = std::max(ar, ma);
    vec pacf = (maxLag > 0) ? sampleYWpacf(yClean, maxLag) : vec();
    vec acf  = (maxLag > 0) ? sampleACF (yClean, maxLag) : vec();
    auto clamp = [](double v, double fb){
        if (!std::isfinite(v)) return fb;
        if (v >  0.85) return  0.85;
        if (v < -0.85) return -0.85;
        return v;
    };
    // Build the p0 vector ourselves so we can hand it straight to estim().
    vec p0(ar + ma + 1, fill::zeros);
    p0(0) = -1.0;
    for (int i = 0; i < ar; ++i)
        p0(1 + i) = clamp(pacf(i), 0.1);
    for (int j = 0; j < ma; ++j)
        p0(1 + ar + j) = clamp(acf(j), -0.1);
    if (ar > 0 && ma > 0){
        double a = p0(1);
        double m = p0(1 + ar);
        if (std::abs(a - m) < 0.1){
            double offset = (m > 0.0) ? -0.2 : 0.2;
            p0(1 + ar) = clamp(m + offset, -0.1);
        }
    }

    // Construct ARMA model (state-space ARMA from src/ARMAmodel.h).
    ARMAmodel sys(ssIn, ar, ma);

    // ARMAmodel constructor sets userModel + userInputs but NOT llikFUN —
    // SSmodel::estim() expects llikFUN to be wired before being called.
    // Use the non-augmented `llik` (ARMA is stationary, no diffuse phase).
    SSinputs sysIn = sys.SSmodel::getInputs();
    sysIn.llikFUN = llik;
    sysIn.p0      = p0;
    // The ARMAinputs handle in sysIn.userInputs was set to a member of `sys`
    // by its constructor — that pointer must outlive estim(), which it
    // does since `sys` is still on this frame.
    sys.SSmodel::setInputs(sysIn);

    // Run the same BFGS the PTS engine uses.
    sys.SSmodel::estim();

    // Pull results back out.
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
    // (the same polyStationary the engine applies inside armaMatrices).
    NumericVector coef(ar + ma);
    if (ar > 0){
        vec ARp = sysOut.p.rows(1, ar);
        polyStationary(ARp);
        for (int i = 0; i < ar; ++i) coef[i] = -ARp(i);     // (1 - φ·B) sign
    }
    if (ma > 0){
        vec MAp = sysOut.p.rows(ar + 1, ar + ma);
        polyStationary(MAp);
        for (int j = 0; j < ma; ++j) coef[ar + j] = MAp(j); // (1 + θ·B) sign
    }
    double sigma2 = sysOut.innVariance;

    return List::create(
        Named("logLik")  = LL,
        Named("AIC")     = aic,
        Named("AICc")    = aicc,
        Named("BIC")     = bic,
        Named("BICc")    = bicc,
        Named("coef")    = coef,
        Named("sigma2")  = sigma2,
        Named("succeed") = std::isfinite(LL));
}
