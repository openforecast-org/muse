#include <RcppArmadillo.h>
#include <iostream>
#include <cmath>
// [[Rcpp::depends(RcppArmadillo)]]

using namespace Rcpp;

/* # Simulator of ctll */
// [[Rcpp::export]]
arma::mat ctllSim(arma::mat states, arma::mat const &noiseEta,
                  arma::mat const &noiseEpsilon, bool const &logValue){
    int nsim = noiseEpsilon.n_cols;
    int obs = noiseEpsilon.n_rows;

    for(unsigned int i=1; i<obs; i=i+1){
        states.row(i) = states.row(i-1) + noiseEta.row(i);
    }

    arma::mat yValues = states + noiseEpsilon;

    if(logValue){
        yValues = exp(yValues);
    }

    return yValues;
}
