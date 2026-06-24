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
            SEXP nsims, SEXP seeds, SEXP lambdaLowers,
            SEXP aEndIns, SEXP PEndIns, SEXP innVarIns, SEXP betaAugIns){

    // --- Marshall SEXP -> MuseInputs (no engine logic in this file) ---
    MuseInputs in;
    NumericVector yr(ys), periodsr(periodss), rhosr(rhoss), p0r(p0s), TVPr(TVPs);
    NumericMatrix ur(us);
    NumericVector aEndr(aEndIns), betaAugr(betaAugIns);
    NumericMatrix PEndr(PEndIns);

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
    in.lambdaLower      = as<double>(lambdaLowers);   // -Inf sentinel = no bound
    // Terminal-state cache (empty aEnd => no cache => full re-filter).
    in.aEndIn           = vec(aEndr.begin(), aEndr.size(), false);
    in.PEndIn           = mat(PEndr.begin(), PEndr.nrow(), PEndr.ncol(), false);
    in.innVarIn         = as<double>(innVarIns);
    in.betaAugIn        = vec(betaAugr.begin(), betaAugr.size(), false);

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
        Named("nInitial") = out.nInitial,
        Named("criteria") = out.criteria);

    if (out.hasValidate){
        output("table")        = out.table;
        output("v")            = out.v;
        output("covp")         = out.covp;
        output("coef")         = out.coef;
        output("typeOutliers") = out.typeOutliers;
    }
    if (out.hasFilter){
        output("v")          = out.v;
    }
    if (out.hasComponents){
        output("comp")      = out.comp;
        output("m")         = out.m;
        output("compNames") = out.compNames;
    }
    if (out.hasSimulate){
        output("simPaths") = out.simPaths;   // h x nsim, original-scale
    }
    if (out.hasInitCache){
        output("aEnd")    = out.aEndOut;
        output("PEnd")    = out.PEndOut;
        output("innVar")  = out.innVarOut;
        output("betaAug") = out.betaAugOut;
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
    (void)criterion_;     // unused -- wrapper returns all four ICs, R picks
    NumericVector yr(ys);
    vec y(yr.begin(), yr.size(), false);
    IntegerVector arOrdersR(arOrders_), maOrdersR(maOrders_), armaLagsR(armaLags_);
    ivec arOrders(arOrdersR.size()), maOrders(maOrdersR.size()),
         armaLags(armaLagsR.size());
    for (R_xlen_t i = 0; i < arOrdersR.size(); ++i) arOrders(i) = arOrdersR[i];
    for (R_xlen_t i = 0; i < maOrdersR.size(); ++i) maOrders(i) = maOrdersR[i];
    for (R_xlen_t i = 0; i < armaLagsR.size(); ++i) armaLags(i) = armaLagsR[i];

    ArmaScoreOutput res;
    runArmaScore(y, arOrders, maOrders, armaLags, res);
    if (!res.succeed)
        return List::create(Named("succeed") = false);
    return List::create(
        Named("logLik")    = res.logLik,
        Named("AIC")       = res.AIC,
        Named("AICc")      = res.AICc,
        Named("BIC")       = res.BIC,
        Named("BICc")      = res.BICc,
        Named("coef")      = NumericVector(res.coef.begin(), res.coef.end()),
        Named("sigma2")    = res.sigma2,
        Named("residuals") = NumericVector(res.residuals.begin(), res.residuals.end()),
        Named("succeed")   = true);
}
