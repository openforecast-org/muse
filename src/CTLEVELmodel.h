// #include <iostream>
// #include <math.h>
// #include <string.h>
// #include <armadillo>
// using namespace arma;
// using namespace std;
// #include "DJPTtools.h"
// #include "optim.h"
// #include "stats.h"
// #include "SSpaceTV.h"

/**************************
 * Model CLASS Continuous Time level model
 ***************************/
struct CTLEVELmodel{
    string obsEq = "stock"; // "stock" or "flow"
    mat delta;
};
class CTLEVELclass : public SSmodel{
public:
    bool errorExit = false;
    string obsEq = "stock"; // "stock" or "flow"
    // SSmodel mSS;
    CTLEVELmodel userInputs;
    // Constructors
    CTLEVELclass(){};
    CTLEVELclass(SSinputs, vec, mat, vec, string, bool, vec, bool);
    // Rest of methods
    // void CTLEVELmatrices(vec, SSmatrix*, void*);
    // void estim(){mSS.estim();};
    // void forecast(){mSS.forecast();};
    // void filter(){mSS.filter();};
    // void smooth(){mSS.smooth(false);};
    // void validate(){mSS.validate(true, 2);};
};
/**************************
 * Functions declarations
 ***************************/
void CTLEVEL(vec, mat, vec, string, bool, vec, bool);
// CTLEVELclass preProcess(vec, mat, vec, string, bool, vec);
void CTLEVELmatrices(vec, SSmatrix*, void*);
/**************************
* Functions implementations
***************************/
// Constructor
CTLEVELclass::CTLEVELclass(SSinputs data, vec y, mat u, vec t, string obsEq,
                           bool verbose, vec p0, bool cllik) : SSmodel(data){
    this->obsEq = obsEq;
    bool errorExit = false;
    // NaNs in u
    if (u.n_rows > 0 && u.has_nonfinite()){
        // MUTE checks - R doesn't like them
        printf("%s", "ERROR: missing values not allowed in input variables!!!\n");
        this->errorExit = true;
    }
    if (u.n_rows > u.n_cols){
        u = u.t();
    }
    // Checking length of t with respect to u
    if (u.n_cols > 0 && u.n_rows != t.n_rows){
        printf("%s", "ERROR: Time index length and inputs length should be equal!!!\n");
        this->errorExit = true;
    }
    // Checking length of t with respect to y
    if (y.n_rows > t.n_rows){
        printf("%s", "ERROR: Time index length should be equal or greater to output length!!!\n");
        this->errorExit = true;
    }
    // obsEq options
    lower(obsEq);
    if (obsEq[0] != 's' && obsEq[0] != 'f'){
        printf("%s", "ERROR: obsEq input should be either \"stock\" or \"flow\"!!!\n");
        this->errorExit = true;
    }
    if (any(p0 < 0.0)){
        printf("%s", "ERROR: Initial parameter values should be positive!!!\n");
        this->errorExit = true;
    }
    if (this->errorExit)
        return;
    // Concentrated likelihood on/off!!!!!!!
    this->SSmodel::inputs.cLlik = cllik;   // Concentrated likelihood on/off
    // Initial estimate
    if (p0.n_elem == 0) {
        if (!cllik) {
            // p0.resize(1).fill(-2.3);
        // else {
            p0.resize(2);
            p0(0) = -2.3;
            p0(1) = -1.15;
        }
    } else if (p0.n_elem == 1) {
        p0 = log(p0) / 2;
        if (!cllik) {
            vec aux(2);
            aux(0) = p0(0);
            aux(1) = -1.15;
            p0.resize(2);
            p0 = aux;
        }
    } else if (p0.n_rows >= 2)
        p0 = log(p0.rows(0, !cllik)) / 2;
    int h = t.n_rows - y.n_rows;
    // Setting up system matrices for CTL model
    // SSinputs input = this->mSS.getInputs();
    // SSmatrix sys = input.system;
    this->SSmodel::inputs.exact = false;
    if (u.n_rows > 0){
        this->SSmodel::inputs.augmented = true;
        this->SSmodel::inputs.llikFUN = llikAug;
    } else {
        this->SSmodel::inputs.augmented = false;
        this->SSmodel::inputs.llikFUN = llik;
    }
    // CTLEVELclass mClass(obsEq);
    // mClass.errorExit = errorExit;
    mat delta(t.n_elem, 1);
    delta(0, 0) = 1.0;
    delta.rows(1, t.n_elem - 1) = diff(t);
    SSmatrix sys;
    if (!errorExit){
        if (obsEq[0] == 's'){
            sys.T.set_size(1, 1); sys.T(0, 0) = 1.0;
            sys.R = sys.T;
            sys.Q = delta;
            sys.Z = sys.T;
            sys.C = sys.T;
            sys.H = sys.T;
            sys.D = sys.Gam = sys.S = 0.0;
            // sys.delta = delta;
        } else{
            // printf("%s", "Not done yet!!!!!");
            sys.T.set_size(2 * (t.n_elem), 2);
            sys.T.fill(0.0);
            sys.T.col(0).fill(1.0);
            sys.R = {{1.0, 0.0, 0.0}, {0.0, 1.0, 1.0}};
            sys.Z = {0, 1};
            sys.C = {0.0};
            sys.H = {0.0};
            sys.Q.set_size(3 * t.n_elem, 3);
            sys.Q.fill(0.0);
        }
        errorExit = checkSystem(sys);
        // Creating model object
        this->SSmodel::inputs.h = h;
        this->SSmodel::inputs.y = y;
        this->SSmodel::inputs.u = u;
        this->SSmodel::inputs.verbose = verbose;
        this->SSmodel::inputs.system = sys;
        this->SSmodel::inputs.p0 = p0;
        // CTLEVELmodel userInputs;
        this->userInputs.delta.resize(delta.size());
        this->userInputs.delta = delta;
        this->userInputs.obsEq = obsEq;
        this->SSmodel::inputs.userInputs = &this->userInputs;
        this->SSmodel::inputs.userModel = CTLEVELmatrices;
    }
    // Creating class
    // SSmodel mSS = SSmodel(input);
    // mSS.setInputs(input);
}
// Main function
void CTLEVEL(vec y, mat u, vec t, string obsEq, bool verbose, vec p0, bool cllik){
    // Building standard SS model
    SSinputs mSS;
    CTLEVELclass mClass(mSS, y, u, t, obsEq, verbose, p0, cllik);
    // mClass = preProcess(y, u, t, obsEq, verbose, p0);
    mClass.estim();
    mClass.filter();
    mClass.validate(true, 2);
}
// matrices del sistema
void CTLEVELmatrices(vec p, SSmatrix* model, void* userInputs){
    // CTLEVELmodel* funInputs = (CTLEVELmodel*)userInputs;
    CTLEVELmodel* funInputs = static_cast<CTLEVELmodel*>(userInputs);
    if (funInputs->obsEq[0] == 's'){
        model->Q = exp(2 * p(0)) * funInputs->delta;
        if (p.n_rows > 1)
            model->H = exp(2 * p(1));
    } else{
        uvec indC(1, fill::zeros);
        model->T(regspace<uvec>(1, 2, model->T.n_rows - 1), indC) = funInputs->delta;
        uvec indR = regspace<uvec>(0, 3, model->Q.n_rows - 1);
        double varEta = exp(2 * p(0));
        // mat aux(model->n, 1, fill::value(varEta));
        model->Q(indR, indC) = funInputs->delta * varEta;
        mat aux = 0.5 * pow(funInputs->delta, 2) * varEta;
        model->Q(indR, indC + 1) = aux;
        model->Q(indR + 1, indC) = aux;
        model->Q(indR + 1, indC + 1) = (1.0 / 3.0) * pow(funInputs->delta, 3) * varEta;
        model->Q(indR + 2, indC + 2) = funInputs->delta;
        if (p.n_rows > 1)
            model->Q(indR + 2, indC + 2) *= exp(2 * p(1));
    }
}
