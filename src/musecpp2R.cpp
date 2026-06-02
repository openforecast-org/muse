// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;
using namespace std;
using namespace Rcpp;
#include "PTSmodel.h"

// [[Rcpp::export]]
SEXP UCompC(SEXP commands, SEXP ys, SEXP us, SEXP models, SEXP hs, SEXP lambdas, SEXP outliers, SEXP tTests,
            SEXP criterions, SEXP periodss, SEXP rhoss, SEXP verboses, SEXP stepwises, SEXP p0s, SEXP armas, SEXP TVPs,
            SEXP seass, SEXP trendOptionss, SEXP seasonalOptionss, SEXP irregularOptionss){
    string command = CHAR(STRING_ELT(commands, 0));
    NumericVector yr(ys);
    NumericMatrix ur(us);
    string model = CHAR(STRING_ELT(models, 0));
    int h = as<int>(hs);
    double lambda = as<double>(lambdas);
    double outlier = as<double>(outliers);
    bool tTest = as<bool>(tTests);
    string criterion = CHAR(STRING_ELT(criterions, 0));
    NumericVector periodsr(periodss);
    NumericVector rhosr(rhoss);
    bool verbose = as<bool>(verboses);
    bool stepwise = as<bool>(stepwises);
    NumericVector p0r(p0s);
    bool arma = as<bool>(armas);
    NumericVector TVPr(TVPs);
    double seas = as<double>(seass);
    string trendOptions = CHAR(STRING_ELT(trendOptionss, 0));
    string seasonalOptions = CHAR(STRING_ELT(seasonalOptionss, 0));
    string irregularOptions = CHAR(STRING_ELT(irregularOptionss, 0));

    vec y(yr.begin(), yr.size(), false);
    mat u(ur.begin(), ur.nrow(), ur.ncol(), false);
    vec periods(periodsr.begin(), periodsr.size(), false);
    vec rhos(rhosr.begin(), rhosr.size(), false);
    vec p0(p0r.begin(), p0r.size(), false);
    vec TVP(TVPr.begin(), TVPr.size(), false);

    // Correcting dimensions of u (k x n)
    size_t k = u.n_rows;
    size_t n = u.n_cols;
    if (k > n){
        u = u.t();
    }
    if (k == 1 && n == 2){
        u.resize(0);
    }
    int iniObs = periods.n_elem * 2 + 2;
    // Setting inputs
    SSinputs inputsSS;
    BSMmodel inputsBSM;
    if (TVP.n_elem == 1 && TVP(0) == -9999.9)
        TVP.reset();
    // Pre-processing
    bool errorExit = preProcess(y, u, model, h, outlier, criterion, periods, p0, iniObs,
                                trendOptions, seasonalOptions, irregularOptions, TVP, lambda);
    if (errorExit)
        return List::create(Named("model") = "error");
    if (sum(TVP) > 0)
        outlier = 0;
    // End of pre-processing
    // if (command == "estimate"){
    // Missing at beginning
    inputsSS.y = y.rows(iniObs, y.n_elem - 1);
    // } else {
    //         inputsSS.y = y;
    // }
    // mat uIni;
    // if (iniObs > 0 && u.n_rows > 0 && command == "estimate"){
    if (u.n_rows > 0) {
        // Missing at beginning
        inputsSS.u = u.cols(iniObs, u.n_cols - 1);
        // inputsSS.u = u.cols(iniObs, y.n_elem - 1);
        // uIni = u.cols(0, iniObs - 1);
    } else {
        inputsSS.u= u;
    }
    inputsBSM.model = model;
    inputsBSM.periods = periods;
    inputsBSM.rhos = rhos;
    inputsSS.h = h;
    inputsBSM.tTest = tTest;
    inputsBSM.criterion = criterion;
    //if (TVP(0) == -9999.99)
    //    TVP = {};
    inputsBSM.TVP = TVP;
    inputsBSM.MSOE = false;
    inputsBSM.PTSnames = true;
    inputsBSM.trendOptions = trendOptions;
    inputsBSM.seasonalOptions = seasonalOptions;
    inputsBSM.irregularOptions = irregularOptions;
    inputsSS.p0 = p0;
    inputsSS.outlier = outlier;
    inputsSS.verbose = verbose;
    inputsBSM.stepwise = stepwise;
    inputsBSM.arma = arma;
    inputsBSM.seas = seas;
    // BoxCox transformation
    if (lambda == 9999.9)
        lambda = testBoxCox(y, periods);
    inputsBSM.lambda = lambda;
    inputsSS.y = BoxCox(inputsSS.y, inputsBSM.lambda);
    // y = BoxCox(y, inputsBSM.lambda);
    // Building model
    BSMclass sysBSM = BSMclass(inputsSS, inputsBSM);
    // Commands
    SSinputs inputs;
    BSMmodel inputs2;
    // if (command == "estimate"){
    sysBSM.estim(inputsSS.verbose);
    sysBSM.forecast();
    sysBSM.parLabels();
    // Missing at beginning
    if (iniObs > 0){
        inputs = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        // inputs.y = y;
        inputs.y = inputsSS.y;
        inputs.u = u;
        sysBSM.SSmodel::setInputs(inputs);
        sysBSM.setInputs(inputs2);
    }
    inputs = sysBSM.SSmodel::getInputs();
    inputs2 = sysBSM.getInputs();
    if (inputs2.cycle[0] != 'n' && inputs2.cycle != "?"){
        string model1 = inputs2.model, cycle = inputs2.cycle, cycle0 = inputs2.cycle0;
        vec periods = inputs2.periods, rhos = inputs2.rhos;
        modelCorrect(model, cycle, inputsBSM.cycle0, periods, rhos);
        inputs2.model= model1, inputs2.cycle= cycle, inputs2.cycle0= cycle0;
        inputs2.periods = periods, inputs2.rhos = rhos;
        // sysBSM.setInputs(inputs2);
        // Estimating period of cycles (lines 3101 BSMmodel)
        // int nRhos = sum(inputs2.typePar == 1);
        // uword pos = inputs2.nPar(0) + nRhos;
        // uvec ind, ind1;
        // vec pInd;
        // int nPerEstim = sum(inputs2.typePar == 2);
        // if (nPerEstim > 0){
        //     ind = regspace<uvec>(pos, 1, pos + nPerEstim - 1);
        //     pos += nPerEstim;
        //     ind1 = find(inputs2.periods < 0);
        //     pInd = inputs.p(ind);
        //     constrain(pInd, inputs2.cycleLimits.rows(ind1));
        //     periods(ind1) = pInd;
        //     inputs2.periods = periods;
        // }
        sysBSM.setInputs(inputs2);
    }
    // Values to return
    inputs = sysBSM.SSmodel::getInputs();
    inputs2 = sysBSM.getInputs();
    if (!inputs2.succeed) {
        return List::create(Named("model") = "error");
    }
    // Converting back to R
    List output = List::create(Named("model") = inputs2.model,
                               Named("yFor") = inputs.yFor,
                               Named("h") = inputs.h,
                               Named("yForV") = inputs.FFor,
                               Named("estimOk") = inputs.estimOk,
                               Named("lambda") = inputs2.lambda,
                               Named("periods") = inputs2.periods,
                               Named("rhos") = inputs2.rhos,
                               Named("p") = inputs.p,
                               Named("p0") = inputs2.p0Return,
                               Named("parNames") = inputs2.parNames,
                               Named("ns") = sum(inputs2.ns),
                               Named("criteria") = inputs.criteria);
    if (command == "validate" || command == "all") {
        sysBSM.validate(false);
        // Values to return
        inputs = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        output("table") = inputs.table;
        output("v") = inputs.v;
        output("covp") = inputs.covp;
        output("coef") = inputs.coef;
    }
    if (command == "filter" || command == "smooth" || command == "disturb" || command == "all"){
        // sysBSM.initMatricesBsm(inputs2.periods, inputs2.rhos, inputs2.trend, inputs2.cycle, inputs2.seasonal, inputs2.irregular);
        // sysBSM.setSystemMatrices();
        if (command != "all")
            sysBSM.validate(false);
        if (command == "filter"){
            sysBSM.filter();
        } else if (command == "smooth") {
            sysBSM.smooth(false);
        } else if (command == "disturb") {
            sysBSM.disturb();
        }
        inputs = sysBSM.SSmodel::getInputs();
        inputs2 = sysBSM.getInputs();
        string statesN = stateNames(inputs2);
        if (command == "disturb"){
            uvec missing = find_nonfinite(inputs.y);
            inputs.eta.cols(missing).fill(datum::nan);
            inputs2.eps(missing).fill(datum::nan);
        }
        // Nans at very beginning
        if (iniObs > 0 && command != "disturb"){
            uvec missing = find_nonfinite(inputs.y.rows(0, iniObs));
            mat P = inputs.P.cols(0, iniObs);
            sysBSM.interpolate(iniObs);
            if (command == "filter"){
                sysBSM.filter();
            } else if (command == "smooth"){
                sysBSM.smooth(false);
            }
            inputs = sysBSM.SSmodel::getInputs();
            inputs.P.cols(0, iniObs) = P;
            inputs.v(missing).fill(datum::nan);
        }
        output("a") = inputs.a;
        output("P") = inputs.P;
        output("v") = inputs.v;
        output("ns") = sum(inputs2.ns);
        output("yFitV") = inputs.F;
        output("yFit") = inputs.yFit;
        output("eps") = inputs2.eps;
        output("eta") = inputs.eta;
        output("stateNames") = statesN;
    }
    if (command == "components" || command == "all"){
        // sysBSM.initMatricesBsm(inputs2.periods, inputs2.rhos, inputs2.trend, inputs2.cycle, inputs2.seasonal, inputs2.irregular);
        // sysBSM.setSystemMatrices();
        sysBSM.components();
        inputs2 = sysBSM.getInputs();
        string compNames = inputs2.compNames;
        // Nans at very beginning
        if (iniObs > 0){
            inputs = sysBSM.SSmodel::getInputs();
            uvec missing = find_nonfinite(inputs.y.rows(0, iniObs));
            //vec ytrun = inputs.y.rows(0, iniObs);
            mat P = inputs2.compV.cols(0, iniObs);
            sysBSM.interpolate(iniObs);
            sysBSM.components();
            inputs2 = sysBSM.getInputs();
            inputs2.compV.cols(0, iniObs) = P;
            // Setting irregular to nan
            uvec rowI(1); rowI(0) = 0;
            if (compNames.find("Level") != string::npos)
                rowI++;
            if (compNames.find("Slope") != string::npos)
                rowI++;
            if (compNames.find("Seasonal") != string::npos)
                rowI++;
            if (compNames.find("Irr") != string::npos ||
                compNames.find("ARMA") != string::npos)
                inputs2.comp.submat(rowI, missing).fill(datum::nan);
        }
        // Values to return
        //inputs2 = sysBSM.getInputs();
        // Converting back to R
        output("comp") = inputs2.comp;
        output("compV") = inputs2.compV;
        output("m") = inputs2.comp.n_rows;
        output("compNames") = compNames;
    }
    return output;
}
