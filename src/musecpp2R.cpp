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
        Named("lambda")   = out.lambda,
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
