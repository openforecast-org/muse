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
#include "muse_compat.h"   // Rprintf shim under -DMUSE_PYTHON_BUILD (no-op under R)
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
    // Lower bound on Box-Cox lambda for the joint-BFGS estimation
    // path (lambda == 9999.9).  -inf = unbounded; finite value =
    // R-side guard (set to 1e-10 when y has zeros).  Plumbed straight
    // into SSinputs::lambdaLower.
    double       lambdaLower = -arma::datum::inf;
    // Terminal-state cache (forecastOnly): when aEndIn is non-empty, the
    // engine reuses it (and PEndIn / innVarIn / betaAugIn) instead of
    // re-filtering the whole series -- so forecast.pts / predict() are O(h)
    // not O(n*m^3).  betaAugIn carries the augmented-KF state (initial states
    // + regressor coefficients) so xreg models can be forecast from the cache
    // too (adam-style: the fitted object stores everything forecasting needs).
    arma::vec    aEndIn;
    arma::mat    PEndIn;
    double       innVarIn = -1.0;
    arma::vec    betaAugIn;
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
    double                  objFunValue = 0.0;        // concentrated objective from last llik() call
    arma::vec               periods;
    arma::vec               rhos;
    arma::vec               p;
    arma::vec               p0Return;
    std::vector<std::string> parNames;
    int                     ns = 0;
    arma::vec               criteria;

    // Terminal-state cache (forecastOnly): the absolute-scale terminal state
    // at the converged params, so a fitted object can store it and later
    // forecasts skip the full-series re-filter.
    bool                    hasInitCache = false;
    arma::vec               aEndOut;
    arma::mat               PEndOut;
    double                  innVarOut = 1.0;
    arma::vec               betaAugOut;

    // validate / all
    bool                    hasValidate = false;
    std::vector<std::string> table;
    arma::vec               v;
    arma::mat               covp;
    arma::vec               coef;

    // innovations (part of "all").  Only `v` is consumed by the R/Python
    // front-ends; the filtered states / smoothed disturbances the engine
    // also computes (a, P, eps, eta, yFit, …) are not surfaced.
    bool                    hasFilter = false;

    // components / all
    bool                    hasComponents = false;
    arma::mat               comp;
    int                     m = 0;
    std::string             compNames;

    // simulate
    bool                    hasSimulate = false;
    arma::mat               simPaths;    // h x nsim, on the original (post-invBoxCox) scale

    // outlier detection (populated by `all` when in.outlier > 0).  Each
    // row is (type, time): type ∈ {0 = AO, 1 = LS, 2 = SC}, time is the
    // 0-based observation index of the detected event.  Zero-row matrix
    // when in.outlier == 0 or nothing was detected.
    arma::mat               typeOutliers;
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
    const bool skipEstim = (command == "forecastOnly" || command == "simulate" ||
                            command == "simulateInit");
    vec userParams;
    if (skipEstim){
        userParams   = in.p0;
        inputsSS.p0  = vec({-9999.9});
        // Terminal-state cache: when supplied, setEstimatedParams() reuses it
        // and skips the full-series re-filter.
        if (in.aEndIn.n_elem > 0){
            inputsSS.aEnd        = in.aEndIn;
            inputsSS.PEnd        = in.PEndIn;
            inputsSS.innVariance = in.innVarIn;
            inputsSS.betaAug     = in.betaAugIn;   // xreg coefs + initial states
        }
    } else {
        inputsSS.p0  = in.p0;
    }
    inputsSS.outlier  = in.outlier;
    inputsSS.verbose  = in.verbose;
    inputsBSM.stepwise = in.stepwise;
    inputsBSM.arma     = in.armaFlag;
    inputsBSM.seas     = in.seas;

    // y_raw retains the untransformed trimmed series so llik() can re-BoxCox
    // per BFGS step when jointly estimating lambda.
    inputsSS.y_raw = inputsSS.y;
    inputsSS.lambdaLower = in.lambdaLower;   // -inf when no R-side guard
    if (in.lambda == 9999.9){
        // Joint estimation: lambda is the last element of the BFGS p vector.
        // llik() re-applies BoxCox(y_raw, p.back()) at every evaluation so
        // we leave inputsSS.y untransformed here.
        double lambda0           = testBoxCox(in.y, in.periods);
        // Lift the warm-start above the user-supplied lower bound
        // (no-op when lambdaLower = -inf).
        if (lambda0 < in.lambdaLower) lambda0 = in.lambdaLower;
        inputsBSM.lambda         = lambda0;
        inputsSS.lambda          = lambda0;
        inputsBSM.profileLambda  = true;   // persistent "joint lambda" flag
        inputsSS.estimateLambda  = true;
        inputsBSM.estimateLambda = true;
        // inputsSS.y stays at y_raw
    } else {
        inputsBSM.profileLambda  = false;
        inputsSS.estimateLambda  = false;
        inputsBSM.estimateLambda = false;
        inputsBSM.lambda         = in.lambda;
        inputsSS.lambda          = in.lambda;
        inputsSS.y               = BoxCox(inputsSS.y, inputsBSM.lambda);
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
    // (PTS never uses stochastic cycles: the cycle slot is always "none",
    // so the old cycle-string correction step here was unreachable.)
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
    out.objFunValue     = inputs.objFunValue;
    out.periods  = inputs2.periods;
    out.rhos     = inputs2.rhos;
    out.p        = inputs.p;
    out.p0Return = inputs2.p0Return;
    out.parNames = inputs2.parNames;
    out.ns       = static_cast<int>(sum(inputs2.ns));
    out.criteria = inputs.criteria;

    // -- terminal-state cache (forecastOnly) --
    // After forecast(), inputs.{aEnd,PEnd,innVariance} hold the absolute-scale
    // terminal state at the converged params.  Surface it so the R/Python
    // front-ends can store it on the fitted object and have later forecasts
    // reuse it (skipping the full-series re-filter).  Only meaningful for the
    // forecastOnly command (the "all" path's state is on the concentrated
    // scale and is not surfaced as a cache).
    if (command == "forecastOnly"){
        out.hasInitCache = true;
        out.aEndOut    = inputs.aEnd;
        out.PEndOut    = inputs.PEnd;
        out.innVarOut  = inputs.innVariance;
        out.betaAugOut = inputs.betaAug;
    }

    // -- validate (runs as part of "all") --
    if (command == "all"){
        sysBSM.validate(false);
        inputs  = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        out.hasValidate = true;
        out.table = inputs.table;
        out.v     = inputs.v;
        out.covp  = inputs.covp;
        out.coef  = inputs.coef;
        // Outlier detection populated by estimOutlier() — empty matrix
        // when in.outlier == 0 or nothing was detected.
        out.typeOutliers = inputs2.typeOutliers;
    }

    // -- innovations (part of "all") --
    // Only `v` is consumed downstream (R derives residuals from it); the
    // missing-value masking below is the reason this block is kept separate
    // from the components extraction.
    if (command == "all"){
        inputs  = sysBSM.SSmodel::getInputs();
        if (iniObs > 0){
            uvec missing = find_nonfinite(inputs.y.rows(0, iniObs));
            mat P = inputs.P.cols(0, iniObs);
            sysBSM.interpolate(iniObs);
            inputs = sysBSM.SSmodel::getInputs();
            inputs.P.cols(0, iniObs) = P;
            inputs.v(missing).fill(datum::nan);
        }
        out.hasFilter = true;
        out.v         = inputs.v;
    }

    // -- components (part of "all") --
    if (command == "all"){
        sysBSM.components();
        inputs2 = sysBSM.getInputs();
        string compNames = inputs2.compNames;
        if (iniObs > 0){
            inputs = sysBSM.SSmodel::getInputs();
            uvec missing = find_nonfinite(inputs.y.rows(0, iniObs));
            sysBSM.interpolate(iniObs);
            sysBSM.components();
            inputs2 = sysBSM.getInputs();
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
        out.m         = static_cast<int>(inputs2.comp.n_rows);
        out.compNames = compNames;
    }

    // -- simulate / simulateInit --
    // Forward-simulate sample paths from the fitted state-space model.
    // setEstimatedParams() above already populated the system matrices,
    // the terminal state aEnd, and (via augmented KF) the initial state
    // betaAug via a single llik pass.  We propagate
    //   a_{t+1} = T a_t + R eta_t        eta ~ N(0, Q)
    //   y_t    = Z a_t + eps_t           eps ~ N(0, H)
    // and apply invBoxCox(., lambda) on the way out so the R wrapper sees
    // original-scale paths.  Q is treated as diagonal (BSM components are
    // independent), which is robust to zero variances.
    //
    // The two commands differ only in which state vector seeds each path:
    //   "simulate"     — terminal aEnd; paths are forward forecasts.
    //   "simulateInit" — initial betaAug (first ns elements); paths are
    //                    in-sample replays starting at t = 0.
    if (command == "simulate" || command == "simulateInit"){
        if (in.seed != 0) arma_rng::set_seed(in.seed);
        SSinputs sim = sysBSM.SSmodel::getInputs();
        const int h    = in.h;
        const int nsim = std::max(1, in.nsim);
        mat T  = sim.system.T;
        mat R  = sim.system.R;
        mat Q  = sim.system.Q;
        mat Z  = sim.system.Z.row(0);
        double H = sim.system.H(0, 0);
        vec seedState;
        if (command == "simulateInit"){
            // Run the smoother once so data->a holds the full raw state
            // vector at every t.  The smoothed state at t = 1 (= column
            // 0 of data->a) is the right seed for an in-sample replay.
            // Cheaper than re-fitting and avoids depending on an R-side
            // state handover that can't see the engine's full state
            // vector (R sees only the aggregated component output).
            sysBSM.SSmodel::smooth(false);
            sim = sysBSM.SSmodel::getInputs();
            if (sim.a.n_cols >= 1 && sim.a.n_rows == sim.aEnd.n_elem){
                seedState = sim.a.col(0);
            } else if (sim.betaAug.n_elem >= sim.aEnd.n_elem){
                seedState = sim.betaAug.head(sim.aEnd.n_elem);
            } else {
                seedState = arma::zeros<vec>(sim.aEnd.n_elem);
            }
        } else {
            seedState = sim.aEnd;
        }
        vec sdQ(Q.n_rows);
        for (uword i = 0; i < Q.n_rows; ++i)
            sdQ(i) = std::sqrt(std::max(0.0, Q(i, i)));
        double sdH = std::sqrt(std::max(0.0, H));
        mat paths(h, nsim);
        for (int s = 0; s < nsim; ++s){
            vec a = seedState;
            for (int t = 0; t < h; ++t){
                double y_t = as_scalar(Z * a) + sdH * randn();
                paths(t, s) = y_t;
                vec eta(R.n_cols);
                for (uword i = 0; i < R.n_cols && i < sdQ.n_elem; ++i)
                    eta(i) = sdQ(i) * randn();
                a = T * a + R * eta;
            }
        }
        out.simPaths = invBoxCoxMat(paths, inputs2.lambda);
        out.hasSimulate = true;
    }
}

// ---------------------------------------------------------------------------
// runArmaScore: standalone ARMA / SARMA fitter on a residual vector, shared by
// both front-ends (the R `.UCompARMAC` and the Python `ucomp_arma`).  Used by
// the PTS-then-ARMA order selector to score each (p, q)(P, Q) candidate by IC.
// Plain Armadillo I/O, no R / Python types -- the binding layers only marshal.
struct ArmaScoreOutput {
    bool        succeed = false;
    double      logLik = arma::datum::nan;
    double      AIC = arma::datum::nan, AICc = arma::datum::nan;
    double      BIC = arma::datum::nan, BICc = arma::datum::nan;
    double      sigma2 = arma::datum::nan;
    arma::vec   coef;        // AR blocks concatenated, then MA blocks (natural)
    arma::vec   residuals;   // one-step-ahead innovations
};

inline void runArmaScore(arma::vec y, arma::ivec arOrders, arma::ivec maOrders,
                         arma::ivec armaLags, ArmaScoreOutput& out){
    using namespace arma;

    int arFree = (arOrders.n_elem > 0) ? (int)sum(arOrders) : 0;
    int maFree = (maOrders.n_elem > 0) ? (int)sum(maOrders) : 0;
    if (arFree < 0) arFree = 0;
    if (maFree < 0) maFree = 0;

    uvec ok = find_finite(y);
    vec yClean = y(ok);
    if (yClean.n_elem < (uword)std::max(4, arFree + maFree + 2)){
        out.succeed = false;
        return;
    }
    int n_finite = (int)yClean.n_elem;

    SSinputs ssIn;
    ssIn.y          = yClean;
    ssIn.y_raw      = yClean;
    ssIn.lambda     = 1.0;
    ssIn.u          = mat(0, yClean.n_elem);
    ssIn.cLlik      = true;
    ssIn.augmented  = false;
    ssIn.exact      = (arFree == 0);
    ssIn.h          = 0;
    ssIn.verbose    = false;
    ssIn.userInputs = nullptr;
    ssIn.estimateLambda = false;

    if (arFree == 0 && maFree == 0){
        double sigma2 = var(yClean, 1);
        double LL = -0.5 * n_finite *
                    (std::log(2.0 * datum::pi) + std::log(sigma2) + 1.0);
        int k = 1;
        out.logLik = LL;
        out.AIC  = -2.0 * LL + 2.0 * k;
        out.BIC  = -2.0 * LL + std::log((double)n_finite) * k;
        out.AICc = out.AIC + (2.0 * k * (k + 1)) / std::max(1, n_finite - k - 1);
        out.BICc = out.BIC + (std::log((double)n_finite) * k * (k + 1)) /
                             std::max(1, n_finite - k - 1);
        out.sigma2 = sigma2;
        out.residuals = yClean - mean(yClean);
        out.coef = vec();
        out.succeed = true;
        return;
    }

    int maxLag = 0;
    for (uword b = 0; b < arOrders.n_elem; ++b)
        maxLag = std::max(maxLag, static_cast<int>(arOrders(b) * armaLags(b)));
    for (uword b = 0; b < maOrders.n_elem; ++b)
        maxLag = std::max(maxLag, static_cast<int>(maOrders(b) * armaLags(b)));
    vec pacf = (maxLag > 0) ? sampleYWpacf(yClean, maxLag) : vec();
    vec acf  = (maxLag > 0) ? sampleACF (yClean, maxLag) : vec();
    auto clampSeed = [](double v, double fb){
        if (!std::isfinite(v)) return fb;
        if (v >  0.85) return  0.85;
        if (v < -0.85) return -0.85;
        return v;
    };

    vec p0(arFree + maFree + 1, fill::zeros);
    p0(0) = -1.0;
    uword pos = 1;
    for (uword b = 0; b < arOrders.n_elem; ++b){
        int pi = arOrders(b), Li = armaLags(b);
        for (int j = 0; j < pi; ++j){
            int lagIdx = (j + 1) * Li;
            double seed = (lagIdx <= (int)pacf.n_elem) ? pacf(lagIdx - 1) : 0.1;
            p0(pos++) = clampSeed(seed, 0.1);
        }
    }
    uword maStart = pos;
    for (uword b = 0; b < maOrders.n_elem; ++b){
        int qi = maOrders(b), Li = armaLags(b);
        for (int j = 0; j < qi; ++j){
            int lagIdx = (j + 1) * Li;
            double seed = (lagIdx <= (int)acf.n_elem) ? acf(lagIdx - 1) : -0.1;
            p0(pos++) = clampSeed(seed, -0.1);
        }
    }
    if (arOrders.n_elem > 0 && maOrders.n_elem > 0 &&
        arOrders(0) > 0 && maOrders(0) > 0){
        double a = p0(1), m = p0(maStart);
        if (std::abs(a - m) < 0.1){
            double offset = (m > 0.0) ? -0.2 : 0.2;
            p0(maStart) = clampSeed(m + offset, -0.1);
        }
    }

    ARMAmodel sys(ssIn, arOrders, maOrders, armaLags);
    SSinputs sysIn = sys.SSmodel::getInputs();
    sysIn.llikFUN = llik;
    sysIn.p0      = p0;
    sys.SSmodel::setInputs(sysIn);
    sys.SSmodel::estim();

    SSinputs sysOut = sys.SSmodel::getInputs();
    vec criteria = sysOut.criteria;   // {LLIK, AIC, BIC, AICc, BICc}
    if (criteria.n_elem < 5){
        out.succeed = false;
        return;
    }
    out.logLik = criteria(0);
    out.AIC    = criteria(1);
    out.BIC    = criteria(2);
    out.AICc   = criteria(3);
    out.BICc   = criteria(4);

    vec coef(arFree + maFree);
    {
        uword pIn = 1, pOut = 0;
        for (uword b = 0; b < arOrders.n_elem; ++b){
            int pi = arOrders(b);
            if (pi == 0) continue;
            vec block = sysOut.p.rows(pIn, pIn + pi - 1);
            polyStationary(block);
            for (int j = 0; j < pi; ++j) coef(pOut++) = -block(j);
            pIn += pi;
        }
        for (uword b = 0; b < maOrders.n_elem; ++b){
            int qi = maOrders(b);
            if (qi == 0) continue;
            vec block = sysOut.p.rows(pIn, pIn + qi - 1);
            polyStationary(block);
            for (int j = 0; j < qi; ++j) coef(pOut++) = block(j);
            pIn += qi;
        }
    }
    out.coef      = coef;
    out.sigma2    = sysOut.innVariance;
    out.residuals = sysOut.v;
    out.succeed   = std::isfinite(out.logLik);
}

#endif // MUSE_CORE_H
