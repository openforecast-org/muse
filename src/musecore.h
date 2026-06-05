// musecore.h - language-agnostic dispatch layer for the muse engine.
//
// This file knows only about Armadillo + STL types; it deliberately has
// no Rcpp / pybind11 / R / Python dependency.  The R binding lives in
// musecpp2R.cpp; a future Python binding would marshal MuseInputs from
// py::array_t, call runMuseCommand(), and pack MuseOutputs into a py::dict
// without changing this header.
//
// The dispatch covers the same commands UCompC always supported:
//   "all", "estimate", "validate", "filter", "smooth", "disturb",
//   "components", "forecastOnly".
//
// Output fields are populated conditionally; the boolean has* flags on
// MuseOutputs tell the binding which sections to include.

#ifndef MUSE_CORE_H
#define MUSE_CORE_H

#include <string>
#include <vector>
#include <armadillo>
#include "PTSmodel.h"
#include "bcnorm.h"   // C++ analogue of greybox::dbcnorm; not wired into
                      // estimation yet -- available for any code path that
                      // wants the BC-corrected log-density.

struct MuseInputs {
    std::string  command;
    arma::vec    y;
    arma::mat    u;
    std::string  model;
    int          h;
    double       lambda;
    double       outlier;
    bool         tTest;
    std::string  criterion;
    arma::vec    periods;
    arma::vec    rhos;
    bool         verbose;
    bool         stepwise;
    arma::vec    p0;
    bool         armaFlag;
    arma::vec    TVP;
    double       seas;
    std::string  trendOptions;
    std::string  seasonalOptions;
    std::string  irregularOptions;
    // simulate-only inputs (ignored by other commands)
    int          nsim = 1;
    unsigned     seed = 0;       // 0 -> let Armadillo seed from clock
};

struct MuseOutputs {
    // Error sentinel: when true the binding should return {"model": "error"}
    bool isError = false;

    // Always populated on success.
    std::string             model;
    arma::vec               yFor;
    int                     h = 0;
    arma::vec               yForV;
    std::string             estimOk;
    double                  lambda = 0.0;
    bool                    lambdaEstimated = false;  // true iff lambda kept (+1 DoF on R side)
    arma::vec               periods;
    arma::vec               rhos;
    arma::vec               p;
    arma::vec               p0Return;
    std::vector<std::string> parNames;
    int                     ns = 0;
    arma::vec               criteria;

    // validate / all
    bool                    hasValidate = false;
    std::vector<std::string> table;
    arma::vec               v;
    arma::mat               covp;
    arma::vec               coef;

    // filter / smooth / disturb / all
    bool                    hasFilter = false;
    arma::mat               a;
    arma::mat               P;
    arma::mat               yFitV;   // SSinputs::F
    arma::vec               yFit;
    arma::vec               eps;
    arma::mat               eta;
    std::string             stateNamesStr;

    // components / all
    bool                    hasComponents = false;
    arma::mat               comp;
    arma::mat               compV;
    int                     m = 0;
    std::string             compNames;

    // simulate
    bool                    hasSimulate = false;
    arma::mat               simPaths;    // h x nsim, on the original (post-invBoxCox) scale
};

// runMuseCommand: end-to-end dispatch. Mirrors what the old UCompC body
// did, but takes / returns plain structs instead of SEXP.  Safe to call
// from any C++ binding layer (R, Python, etc.).
inline void runMuseCommand(MuseInputs in, MuseOutputs& out){
    using namespace arma;
    using std::string;

    const string& command = in.command;

    // Adjust u dimensions: engine expects k (rows) x n (cols).
    {
        size_t k = in.u.n_rows, n = in.u.n_cols;
        if (k > n) in.u = in.u.t();
        if (k == 1 && n == 2) in.u.resize(0);
    }
    int iniObs = static_cast<int>(in.periods.n_elem) * 2 + 2;

    // Drop the TVP sentinel; the engine expects an empty vector instead.
    if (in.TVP.n_elem == 1 && in.TVP(0) == -9999.9)
        in.TVP.reset();

    // Pre-process: may rewrite model, periods, lambda; iniObs adjusted.
    bool errorExit = preProcess(in.y, in.u, in.model, in.h, in.outlier,
                                in.criterion, in.periods, in.p0, iniObs,
                                in.trendOptions, in.seasonalOptions,
                                in.irregularOptions, in.TVP, in.lambda);
    if (errorExit){
        out.isError = true;
        return;
    }
    if (sum(in.TVP) > 0)
        in.outlier = 0;

    // Build the SS / BSM input bundles the engine consumes.
    SSinputs inputsSS;
    BSMmodel inputsBSM;
    inputsSS.y = in.y.rows(iniObs, in.y.n_elem - 1);
    if (in.u.n_rows > 0){
        inputsSS.u = in.u.cols(iniObs, in.u.n_cols - 1);
    } else {
        inputsSS.u = in.u;
    }
    inputsBSM.model            = in.model;
    inputsBSM.periods          = in.periods;
    inputsBSM.rhos             = in.rhos;
    inputsSS.h                 = in.h;
    inputsBSM.tTest            = in.tTest;
    inputsBSM.criterion        = in.criterion;
    inputsBSM.TVP              = in.TVP;
    inputsBSM.MSOE             = false;
    inputsBSM.PTSnames         = true;
    inputsBSM.trendOptions     = in.trendOptions;
    inputsBSM.seasonalOptions  = in.seasonalOptions;
    inputsBSM.irregularOptions = in.irregularOptions;

    // forecastOnly / simulate: hide the user params from initParBsm (which
    // would crash on zero / tiny variances), stash them, push them in via
    // setEstimatedParams after construction.
    const bool skipEstim = (command == "forecastOnly" || command == "simulate");
    vec userParams;
    if (skipEstim){
        userParams   = in.p0;
        inputsSS.p0  = vec({-9999.9});
    } else {
        inputsSS.p0  = in.p0;
    }
    inputsSS.outlier  = in.outlier;
    inputsSS.verbose  = in.verbose;
    inputsBSM.stepwise = in.stepwise;
    inputsBSM.arma     = in.armaFlag;
    inputsBSM.seas     = in.seas;

    // y_raw retains the untransformed trimmed series in every path so the
    // profile-lambda outer loop (BSMclass::profileLambda) can re-Box-Cox
    // for each candidate lambda without leaning on llik to do it inline.
    inputsSS.y_raw          = inputsSS.y;
    inputsSS.estimateLambda = false;       // joint-lambda path retired
    inputsBSM.estimateLambda = false;
    if (in.lambda == 9999.9){
        // Profile lambda: Brent outer search with per-step inner BFGS.
        // We leave inputsSS.y at y_raw (untransformed) so the constructor's
        // initParBsm is tuned for lambda=1 — the same state the ident sweep's
        // estimUCs sees for non-first candidates (where y_raw is left by the
        // previous snap).  profileLambda's innerFit re-BoxCoxes at every Brent
        // step, so the actual estimation y is always correct.
        // Starting Brent at 1.0 (a neutral anchor) rather than testBoxCox
        // ensures the first BFGS call has a matched (y_raw, p0_from_y_raw)
        // pair, matching ident-sweep quality for the common snap-to-1 case.
        inputsBSM.profileLambda = true;
        inputsBSM.lambda        = 1.0;
        // inputsSS.y intentionally stays at y_raw (no BoxCox here)
    } else {
        inputsBSM.profileLambda = false;
        inputsBSM.lambda        = in.lambda;
        inputsSS.y              = BoxCox(inputsSS.y, inputsBSM.lambda);
    }

    BSMclass sysBSM(inputsSS, inputsBSM);
    if (skipEstim){
        sysBSM.setEstimatedParams(userParams);
    } else {
        sysBSM.estim(inputsSS.verbose);
    }
    sysBSM.forecast();
    sysBSM.parLabels();

    // Restore the full y / u (with iniObs padding) so downstream commands
    // see the original series.
    SSinputs inputs;
    BSMmodel inputs2;
    if (iniObs > 0){
        inputs  = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        inputs.y = inputsSS.y;
        inputs.u = in.u;
        sysBSM.SSmodel::setInputs(inputs);
        sysBSM.setInputs(inputs2);
    }
    inputs  = sysBSM.SSmodel::getInputs();
    inputs2 = sysBSM.getInputs();

    // Cycle bookkeeping.
    if (inputs2.cycle[0] != 'n' && inputs2.cycle != "?"){
        string model1 = inputs2.model, cycle = inputs2.cycle, cycle0 = inputs2.cycle0;
        vec periods = inputs2.periods, rhos = inputs2.rhos;
        modelCorrect(in.model, cycle, inputsBSM.cycle0, periods, rhos);
        inputs2.model = model1; inputs2.cycle = cycle; inputs2.cycle0 = cycle0;
        inputs2.periods = periods; inputs2.rhos = rhos;
        sysBSM.setInputs(inputs2);
    }
    inputs  = sysBSM.SSmodel::getInputs();
    inputs2 = sysBSM.getInputs();
    if (!inputs2.succeed){
        out.isError = true;
        return;
    }

    // -- Always populated --
    out.model    = inputs2.model;
    out.yFor     = inputs.yFor;
    out.h        = inputs.h;
    out.yForV    = inputs.FFor;
    out.estimOk  = inputs.estimOk;
    out.lambda          = inputs2.lambda;
    out.lambdaEstimated = inputs2.lambdaEstimated;
    out.periods  = inputs2.periods;
    out.rhos     = inputs2.rhos;
    out.p        = inputs.p;
    out.p0Return = inputs2.p0Return;
    out.parNames = inputs2.parNames;
    out.ns       = static_cast<int>(sum(inputs2.ns));
    out.criteria = inputs.criteria;

    // -- validate / all --
    if (command == "validate" || command == "all"){
        sysBSM.validate(false);
        inputs  = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        out.hasValidate = true;
        out.table = inputs.table;
        out.v     = inputs.v;
        out.covp  = inputs.covp;
        out.coef  = inputs.coef;
    }

    // -- filter / smooth / disturb / all --
    if (command == "filter" || command == "smooth" || command == "disturb" || command == "all"){
        if (command != "all")
            sysBSM.validate(false);
        if      (command == "filter")  sysBSM.filter();
        else if (command == "smooth")  sysBSM.smooth(false);
        else if (command == "disturb") sysBSM.disturb();

        inputs  = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        string statesN = stateNames(inputs2);
        if (command == "disturb"){
            uvec missing = find_nonfinite(inputs.y);
            inputs.eta.cols(missing).fill(datum::nan);
            inputs2.eps(missing).fill(datum::nan);
        }
        if (iniObs > 0 && command != "disturb"){
            uvec missing = find_nonfinite(inputs.y.rows(0, iniObs));
            mat P = inputs.P.cols(0, iniObs);
            sysBSM.interpolate(iniObs);
            if      (command == "filter") sysBSM.filter();
            else if (command == "smooth") sysBSM.smooth(false);
            inputs = sysBSM.SSmodel::getInputs();
            inputs.P.cols(0, iniObs) = P;
            inputs.v(missing).fill(datum::nan);
        }
        out.hasFilter     = true;
        out.a             = inputs.a;
        out.P             = inputs.P;
        out.v             = inputs.v;
        out.ns            = static_cast<int>(sum(inputs2.ns));
        out.yFitV         = inputs.F;
        out.yFit          = inputs.yFit;
        out.eps           = inputs2.eps;
        out.eta           = inputs.eta;
        out.stateNamesStr = statesN;
    }

    // -- components / all --
    if (command == "components" || command == "all"){
        sysBSM.components();
        inputs2 = sysBSM.getInputs();
        string compNames = inputs2.compNames;
        if (iniObs > 0){
            inputs = sysBSM.SSmodel::getInputs();
            uvec missing = find_nonfinite(inputs.y.rows(0, iniObs));
            mat P = inputs2.compV.cols(0, iniObs);
            sysBSM.interpolate(iniObs);
            sysBSM.components();
            inputs2 = sysBSM.getInputs();
            inputs2.compV.cols(0, iniObs) = P;
            uvec rowI(1); rowI(0) = 0;
            if (compNames.find("Level")    != string::npos) rowI++;
            if (compNames.find("Slope")    != string::npos) rowI++;
            if (compNames.find("Seasonal") != string::npos) rowI++;
            if (compNames.find("Irr")  != string::npos ||
                compNames.find("ARMA") != string::npos)
                inputs2.comp.submat(rowI, missing).fill(datum::nan);
        }
        out.hasComponents = true;
        out.comp      = inputs2.comp;
        out.compV     = inputs2.compV;
        out.m         = static_cast<int>(inputs2.comp.n_rows);
        out.compNames = compNames;
    }

    // -- simulate --
    // Forward-simulate sample paths from the fitted state-space model.
    // setEstimatedParams() above already populated the system matrices and
    // the terminal state aEnd via a single llik pass.  We propagate
    //   a_{t+1} = T a_t + R eta_t        eta ~ N(0, Q)
    //   y_t    = Z a_t + eps_t           eps ~ N(0, H)
    // and apply invBoxCox(., lambda) on the way out so the R wrapper sees
    // original-scale paths.  Q is treated as diagonal (BSM components are
    // independent), which is robust to zero variances.
    if (command == "simulate"){
        if (in.seed != 0) arma_rng::set_seed(in.seed);
        SSinputs sim = sysBSM.SSmodel::getInputs();
        const int h    = in.h;
        const int nsim = std::max(1, in.nsim);
        mat T  = sim.system.T;
        mat R  = sim.system.R;
        mat Q  = sim.system.Q;
        mat Z  = sim.system.Z.row(0);
        double H = sim.system.H(0, 0);
        vec aEnd = sim.aEnd;
        vec sdQ(Q.n_rows);
        for (uword i = 0; i < Q.n_rows; ++i)
            sdQ(i) = std::sqrt(std::max(0.0, Q(i, i)));
        double sdH = std::sqrt(std::max(0.0, H));
        mat paths(h, nsim);
        for (int s = 0; s < nsim; ++s){
            vec a = aEnd;
            for (int t = 0; t < h; ++t){
                double y_t = as_scalar(Z * a) + sdH * randn();
                paths(t, s) = y_t;
                vec eta(R.n_cols);
                for (uword i = 0; i < R.n_cols && i < sdQ.n_elem; ++i)
                    eta(i) = sdQ(i) * randn();
                a = T * a + R * eta;
            }
        }
        // Back-transform to the original scale element-by-element.
        out.simPaths = invBoxCoxMat(paths, inputs2.lambda);
        out.hasSimulate = true;
    }
}

#endif // MUSE_CORE_H
