// [[Rcpp::depends(RcppArmadillo)]] 
#include <RcppArmadillo.h>
using namespace arma;
using namespace std;
using namespace Rcpp;
#include "INTLEVELmodel.h"

// [[Rcpp::export]]
SEXP INTLEVELc(SEXP ys, SEXP us, SEXP hs, SEXP obsEqs,
               SEXP verboses, SEXP p0s, SEXP logTransforms){
        // Translating inputs to armadillo data
        // string command = CHAR(STRING_ELT(commands, 0));
        NumericVector yr(ys);
        mat u;
        if (Rf_isNull(us)){
                u.set_size(0, 0);
        } else {
                NumericMatrix ur(us);
                mat aux(ur.begin(), ur.nrow(), ur.ncol(), false);
                u = aux;
                if (u.n_rows > u.n_cols)
                        u = u.t();
        }
        int h = as<int>(hs);
        string obsEq = CHAR(STRING_ELT(obsEqs, 0));
        NumericVector p0r(p0s);
        bool verbose = as<bool>(verboses);
        bool logTransform = as<bool>(logTransforms);
        // Second step
        vec y(yr.begin(), yr.size(), false);
        vec p0(p0r.begin(), p0r.size(), false);
        // Creating class
        INTLEVELclass mClass(y, u, h, obsEq, verbose, p0, logTransform);
        if (mClass.errorExit)
                return List::create(Named("errorExit") = mClass.errorExit);
        mClass.estim();
        mClass.forecast();
        mClass.smooth();
        mClass.validate();
        SSinputs m = mClass.m.SSmodel::getInputs();
        // Output
        return List::create(Named("p") = m.p,
                            Named("yFor") = m.yFor,
                            Named("yForV") = m.FFor,
                            Named("table") = m.table,
                            Named("comp") = mClass.comp,
                            Named("compV") = mClass.compV,
                            Named("compNames") = mClass.compNames
        );
}
